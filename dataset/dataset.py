from utils.operators import calc_polynomial_coeff_by_points_n_deg, trans_motion2d_to_hips_coord_sys, trans_motion2d_to_hoop_coord_sys, calc_pixels_to_real_units_scaling_factor
from .augmentations import NormalizeMotion, Resize, ToTensor, GaussianNoise, RandomZeroMask
from torch.utils.data import Dataset
import os
import os.path as osp
import numpy as np
from torchvision import transforms
import glob
import pandas as pd
import math
import torch
from torch.utils.data.sampler import RandomSampler, Sampler

# class ExampleImbalancedDatasetSampler(Sampler):
#     """
#
#     """
#
#     def __init__(self, data):
#         self.data = data
#         print(data)
#         exit()
#
#     def __iter__(self):
#         return 0
#
#     def __len__(self):
#         return len(self.data)


class BBFTSBatchSchedulerSampler(Sampler):
    """
    iterate over tasks and provide a balanced batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset.labels_df) for cur_dataset in dataset.datasets])
        # self.smallest_dataset_size = min([len(cur_dataset.labels_df) for cur_dataset in dataset.datasets])

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)
        # return self.batch_size * math.ceil(self.smallest_dataset_size / self.batch_size) #* len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            # if dataset_idx == 0:
                # the first dataset is kept at RandomSampler
            sampler = RandomSampler(cur_dataset)
            # else:
            #     # the second unbalanced dataset is changed
            #     sampler = ExampleImbalancedDatasetSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets
        # epoch_samples = self.smallest_dataset_size * self.number_of_datasets
        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


class BBFTSDataset(Dataset):

    def __init__(self, subset, config):

        self.subset = subset

        self.motions = {}
        self.shot_trajectories = {}

        motion_fpath = osp.join(config.data_dir, osp.join(subset, 'motion'))
        shot_traj_fpath = osp.join(config.data_dir, osp.join(subset, 'shot_trajectories'))
        for curr_motion_name in os.listdir(motion_fpath):
            true_name = osp.splitext(curr_motion_name)[0]

            self.motions[true_name] = np.load(osp.join(motion_fpath, curr_motion_name))
            self.shot_trajectories[true_name] = np.load(osp.join(shot_traj_fpath, curr_motion_name))

        self.n_samples = len(os.listdir(motion_fpath))

        df = pd.read_csv(osp.join(config.data_dir, config.labels_file), header=0)
        self.labels_df = df.loc[df['phase'] == subset]
        self.labels_train_df = df.loc[df['phase'] == 'train']
        self.labels_extras_df = df.loc[df['phase'] == 'extras']

        self.pre_rel_n_frames = config.pre_release_n_frames
        self.post_rel_n_frames = config.post_release_n_frames

        self.hoops_df = pd.read_csv(osp.join(config.data_dir, osp.join(subset, 'hoops_info.csv')), header=0)
        self.hoops_train_df = pd.read_csv(osp.join(config.data_dir, osp.join('train', 'hoops_info.csv')), header=0)
        self.hoops_extras_df = pd.read_csv(osp.join(config.data_dir, osp.join('extras', 'hoops_info.csv')), header=0)
        self.scale_factoring_map = {}
        mean_pose, std_pose = self.get_meanpose(config)

        self.transforms = transforms.Compose([
            NormalizeMotion(mean_pose=mean_pose, std_pose=std_pose),
            Resize(scale=(config.nr_joints * 2, -1)),
            GaussianNoise(mean_pose=0., std_pose=0.01),
            # RandomZeroMask(p=0.03),
            ToTensor()
        ])

        self.objective = 'cls' if subset != 'extras' else 'trj'

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        vid_name = self.labels_df.iloc[idx]['video_name']
        # Create one-hot label vector
        label_idx = self.labels_df.iloc[idx]['label']
        cls_label = torch.from_numpy(np.array([label_idx])).type(torch.long)

        # motion = np.load(osp.join(self.motion_fpath, f'{vid_name}.npy'))
        motion = self.motions[str(vid_name)]
        shot_frame = int(self.labels_df.iloc[idx]['shot_frame'])
        # extend motion by duplication (must have fixed number frames before shot release and after it)
        motion, shot_frame = self.duplicate_pose_by_shot_frame(motion, shot_frame)
        # crop motion around shot release frame
        motion = motion[:, :, shot_frame - self.pre_rel_n_frames:shot_frame + self.post_rel_n_frames]

        hoop_bb = self.hoops_df.loc[self.hoops_df['name'] == f'{vid_name}.npy']['hoop'].item().split(',')
        if vid_name not in self.scale_factoring_map:
            # calculate scaling factor (from pixels to real world units, e.g Feet x alpha)
            scale_factor_x, scale_factor_y = calc_pixels_to_real_units_scaling_factor(motion, hoop_bb, alpha=1.0)
            self.scale_factoring_map[vid_name] = {'X': scale_factor_x, 'Y': scale_factor_y}

        # Transform motion coordinate system to hoop relative
        motion = trans_motion2d_to_hoop_coord_sys(motion.copy(), hoop_bb)
        # Convert units
        motion = motion * np.array([self.scale_factoring_map[vid_name]['X'],
                                    self.scale_factoring_map[vid_name]['Y']]).reshape((1, 2, 1))
        # motion[:, 1, :] *= -1

        motion = self.transforms(motion)

        # Process shot trajectory
        shot_trajectory = self.shot_trajectories[str(vid_name)]
        # Convert shot trajectory coordinate system (relative to first ball in shot trajectory)
        # shot_trajectory[:, 0] = shot_trajectory[:, 0] - shot_trajectory[0][0]
        # shot_trajectory[:, 1] = -1 * (shot_trajectory[:, 1] - shot_trajectory[0][1])
        shot_trajectory = shot_trajectory.T[np.newaxis, ...]

        shot_trajectory = trans_motion2d_to_hoop_coord_sys(shot_trajectory, hoop_bb)

        shot_trajectory = shot_trajectory * np.array([self.scale_factoring_map[vid_name]['X'],
                                                      self.scale_factoring_map[vid_name]['Y']]).reshape((1, 2, 1))

        # Create Label for shot trajectory
        shot_trajectory = shot_trajectory[0, :, :].T
        # shot_trajectory shape is (T, 2)
        # shot_trajectory[:, 1] *= -1
        # first_ball_coords = shot_trajectory[1, :]
        last_ball_coords = shot_trajectory[-1, :]
        high_ball_coords = shot_trajectory[1, :]

        for i in range(2, len(shot_trajectory) - 2):
            if shot_trajectory[i][1] > high_ball_coords[1]:
                high_ball_coords = shot_trajectory[i]

        # trj_labels = torch.Tensor(np.array([first_ball_coords[0], first_ball_coords[1],
        #                                     high_ball_coords[0], high_ball_coords[1],
        #                                     last_ball_coords[0], last_ball_coords[1]]))
        trj_labels = torch.Tensor(np.array([high_ball_coords[0], high_ball_coords[1],
                                            last_ball_coords[0], last_ball_coords[1]]))

        # TODO its only required for visualization
        scale_factors = torch.Tensor(np.array([self.scale_factoring_map[vid_name]['X'],
                                               self.scale_factoring_map[vid_name]['Y']]))

        # Convert units
        sample = {'name': vid_name,
                  'motion': motion,
                  'cls_labels': cls_label,
                  'trj_labels': trj_labels,
                  'scale': scale_factors,
                  'objective': self.objective}

        return sample

    def preproccess(self):
        pass

    def duplicate_pose_by_shot_frame(self, motion, shot_frame):
        if shot_frame <= self.pre_rel_n_frames:
            n_diff = self.pre_rel_n_frames - shot_frame
            shot_frame += n_diff
            first_pose = np.copy(motion[:, :, 0])
            first_pose = first_pose[..., np.newaxis]
            first_pose_matrix = np.repeat(first_pose, n_diff, axis=2)
            motion = np.concatenate((first_pose_matrix, motion), axis=2)

        return motion, shot_frame

    def get_meanpose(self, config):
        meanpose_path = config.meanpose_path
        stdpose_path = config.stdpose_path
        if osp.exists(meanpose_path) and osp.exists(stdpose_path):
            meanpose = np.load(meanpose_path)
            stdpose = np.load(stdpose_path)
        else:
            meanpose, stdpose = self.gen_meanpose(config)
            np.save(meanpose_path, meanpose)
            np.save(stdpose_path, stdpose)
            print("meanpose saved at {}".format(meanpose_path))
            print("stdpose saved at {}".format(stdpose_path))

        return meanpose, stdpose

    def gen_meanpose(self, config):
        # if self.subset == 'extras':
        #     all_paths = sorted(glob.glob(osp.join(config.data_dir, 'extras', 'motion/*.npy')))   # paths to all_motion.npy files
        # else:
        #     all_paths = sorted(glob.glob(osp.join(config.data_dir, 'train', 'motion/*.npy')))   # paths to all_motion.npy files
        all_paths_extras = sorted(glob.glob(osp.join(config.data_dir, 'extras', 'motion/*.npy')))
        all_paths_train = sorted(glob.glob(osp.join(config.data_dir, 'train', 'motion/*.npy')))

        # all_paths.extend(all_paths1)
        all_joints = []
        for path in all_paths_train:
            motion2d = np.load(path)
            curr_vid_name = osp.splitext(osp.basename(path))[0]
            curr_shot_frame = int(self.labels_train_df.loc[self.labels_train_df['video_name'] == int(curr_vid_name)]['shot_frame'].item())
            # extend motion by duplication (must have fixed number frames before shot release and after it)
            motion2d, curr_shot_frame = self.duplicate_pose_by_shot_frame(motion2d, curr_shot_frame)
            # crop motion around shot release frame
            motion2d = motion2d[:, :, curr_shot_frame - self.pre_rel_n_frames:curr_shot_frame + self.post_rel_n_frames]
            # calculate scaling factor (from pixels to real world units, e.g Feet x alpha)
            hoop_bb = self.hoops_train_df.loc[self.hoops_train_df['name'] == f'{curr_vid_name}.npy']['hoop'].item().split(',')
            scale_factor_x, scale_factor_y = calc_pixels_to_real_units_scaling_factor(motion2d, hoop_bb, alpha=1.0)
            self.scale_factoring_map[curr_vid_name] = {'X': scale_factor_x, 'Y': scale_factor_y}
            # Transform motion coordinate system
            motion2d_in_hoop_coords = trans_motion2d_to_hoop_coord_sys(motion2d, hoop_bb)
            # motion2d = trans_motion2d_to_hips_coord_sys(motion2d)
            # Convert units
            motion2d_in_hoop_coords = motion2d_in_hoop_coords * np.array([scale_factor_x, scale_factor_y]).reshape((1, 2, 1))
            # motion2d_in_hoop_coords[:, 1, :] *= -1
            all_joints.append(motion2d_in_hoop_coords)

        for path in all_paths_extras:
            motion2d = np.load(path)
            curr_vid_name = osp.splitext(osp.basename(path))[0]
            curr_shot_frame = int(self.labels_extras_df.loc[self.labels_extras_df['video_name'] == int(curr_vid_name)]['shot_frame'].item())
            # extend motion by duplication (must have fixed number frames before shot release and after it)
            motion2d, curr_shot_frame = self.duplicate_pose_by_shot_frame(motion2d, curr_shot_frame)
            # crop motion around shot release frame
            motion2d = motion2d[:, :, curr_shot_frame - self.pre_rel_n_frames:curr_shot_frame + self.post_rel_n_frames]
            # calculate scaling factor (from pixels to real world units, e.g Feet x alpha)
            hoop_bb = self.hoops_extras_df.loc[self.hoops_extras_df['name'] == f'{curr_vid_name}.npy']['hoop'].item().split(',')
            scale_factor_x, scale_factor_y = calc_pixels_to_real_units_scaling_factor(motion2d, hoop_bb, alpha=1.0)
            self.scale_factoring_map[curr_vid_name] = {'X': scale_factor_x, 'Y': scale_factor_y}
            # Transform motion coordinate system
            motion2d_in_hoop_coords = trans_motion2d_to_hoop_coord_sys(motion2d, hoop_bb)
            # motion2d = trans_motion2d_to_hips_coord_sys(motion2d)
            # Convert units
            motion2d_in_hoop_coords = motion2d_in_hoop_coords * np.array([scale_factor_x, scale_factor_y]).reshape((1, 2, 1))
            # motion2d_in_hoop_coords[:, 1, :] *= -1
            all_joints.append(motion2d_in_hoop_coords)

        all_joints = np.concatenate(all_joints, axis=2)
        meanpose = np.mean(all_joints, axis=2)
        stdpose = np.std(all_joints, axis=2)
        stdpose[np.where(stdpose == 0)] = 1e-9
        return meanpose, stdpose