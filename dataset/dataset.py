from utils.operators import calc_polynomial_coeff_by_points_n_deg, trans_motion2d_to_hips_coord_sys, trans_motion2d_to_hoop_coord_sys, calc_pixels_to_real_units_scaling_factor
from .augmentations import NormalizeMotion, Resize, ToTensor, GaussianNoise, RandomZeroMask
from torch.utils.data import Dataset
import os
import os.path as osp
import numpy as np
import torch
from torchvision import transforms
import glob
import pandas as pd


class BBFTSDataset(Dataset):

    def __init__(self, phase, config):
        # super(BBFTSDataset, self).__init__()
        assert phase in ['train', 'test', 'extras']
        assert config.labels_file.endswith('.csv')
        self.data_fpath = osp.join(config.data_dir, osp.join(phase, 'motion'))
        self.shot_traj_fpath = osp.join(config.data_dir, osp.join(phase, 'shot_trajectories'))
        # self.video_names = os.listdir(self.data_fpath)
        df = pd.read_csv(osp.join(config.data_dir, config.labels_file), header=0)
        self.labels_df = df.loc[df['phase'] == phase]
        self.phase = phase
        self.pre_rel_n_frames = config.pre_release_n_frames
        self.post_rel_n_frames = config.post_release_n_frames

        self.hoops_df = pd.read_csv(osp.join(config.data_dir, osp.join(phase, 'hoops_info.csv')), header=0)
        self.scale_factoring_map = {}
        mean_pose, std_pose = self.get_meanpose(config)

        self.transforms = transforms.Compose([
            NormalizeMotion(mean_pose=mean_pose, std_pose=std_pose),
            Resize(scale=(config.nr_joints * 2, -1)),
            # GaussianNoise(mean_pose=0., std_pose=0.01),
            # RandomZeroMask(p=0.03),
            ToTensor()
        ])

        self.poly_deg = config.poly_deg

    def __len__(self):
        return len(os.listdir(self.data_fpath))

    def __getitem__(self, idx):
        vid_name = self.labels_df.iloc[idx]['video_name']
        # Create one-hot label vector
        label_idx = self.labels_df.iloc[idx]['label']
        label = torch.from_numpy(np.array([label_idx])).type(torch.long)

        vid_fpath = osp.join(self.data_fpath, f'{vid_name}.npy')
        motion = np.load(vid_fpath)
        shot_frame = int(self.labels_df.iloc[idx]['shot_frame'])
        # extend motion by duplication (must have fixed number frames before shot release and after it)
        motion, shot_frame = self.duplicate_pose_by_shot_frame(motion, shot_frame)
        # crop motion around shot release frame
        motion = motion[:, :, shot_frame - self.pre_rel_n_frames:shot_frame + self.post_rel_n_frames]

        hoop_bb = self.hoops_df.loc[self.hoops_df['name'] == f'{vid_name}.npy']['hoop'].item().split(',')
        if vid_name not in self.scale_factoring_map:
            # calculate scaling factor (from pixels to real world units, e.g Feet x alpha)
            scale_factor_x, scale_factor_y = calc_pixels_to_real_units_scaling_factor(motion, hoop_bb, alpha=0.4)
            self.scale_factoring_map[vid_name] = {'X': scale_factor_x, 'Y': scale_factor_y}

        # Transform motion coordinate system
        motion = trans_motion2d_to_hoop_coord_sys(motion, hoop_bb)
        # Convert units
        motion = motion * np.array([self.scale_factoring_map[vid_name]['X'], self.scale_factoring_map[vid_name]['Y']]).reshape((1, 2, 1))

        motion = self.transforms(motion)

        shot_trajectory = np.load(osp.join(self.shot_traj_fpath, f'{vid_name}.npy'))
        shot_traj_len = len(shot_trajectory)
        shot_trajectory = shot_trajectory.T[np.newaxis, ...]
        shot_trajectory = trans_motion2d_to_hoop_coord_sys(shot_trajectory, hoop_bb)
        shot_trajectory = shot_trajectory * np.array([self.scale_factoring_map[vid_name]['X'], self.scale_factoring_map[vid_name]['Y']]).reshape((1, 2, 1))
        n_ball_samples = min(shot_traj_len, 10)
        shot_trajectory = shot_trajectory[0, :, ::(shot_traj_len // n_ball_samples)]
        shot_trajectory = shot_trajectory[:, :10]
        # assert shot_trajectory.shape[-1] == 10
        shot_trajectory = shot_trajectory - shot_trajectory[:, 0].reshape(2, -1)
        # shot_trajectory shape is (2, T)
        shot_trajectory = calc_polynomial_coeff_by_points_n_deg(shot_trajectory[0, :], shot_trajectory[1, :], deg=self.poly_deg)
        # shot_trajectory = np.polyfit(shot_trajectory[0, :], shot_trajectory[1, :], 3)
        shot_trajectory = torch.Tensor(shot_trajectory)
        # motion = trans_motion2d_to_hips_coord_sys(motion)
        # Convert units
        sample = {'name': vid_name, 'motion': motion, 'label': label, 'shot_trajectory': shot_trajectory}

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
        if config.in_pretrain:
            all_paths = sorted(glob.glob(osp.join(config.data_dir, 'extras', 'motion/*.npy')))   # paths to all_motion.npy files
        else:
            all_paths = sorted(glob.glob(osp.join(config.data_dir, 'train', 'motion/*.npy')))   # paths to all_motion.npy files

        all_joints = []
        for path in all_paths:
            motion2d = np.load(path)
            curr_vid_name = osp.splitext(osp.basename(path))[0]
            curr_shot_frame = int(self.labels_df.loc[self.labels_df['video_name'] == int(curr_vid_name)]['shot_frame'].item())
            # extend motion by duplication (must have fixed number frames before shot release and after it)
            motion2d, curr_shot_frame = self.duplicate_pose_by_shot_frame(motion2d, curr_shot_frame)
            # crop motion around shot release frame
            motion2d = motion2d[:, :, curr_shot_frame - self.pre_rel_n_frames:curr_shot_frame + self.post_rel_n_frames]
            # calculate scaling factor (from pixels to real world units, e.g Feet x alpha)
            hoop_bb = self.hoops_df.loc[self.hoops_df['name'] == f'{curr_vid_name}.npy']['hoop'].item().split(',')
            scale_factor_x, scale_factor_y = calc_pixels_to_real_units_scaling_factor(motion2d, hoop_bb, alpha=0.4)
            self.scale_factoring_map[curr_vid_name] = {'X': scale_factor_x, 'Y': scale_factor_y}
            # Transform motion coordinate system
            motion2d_v2 = trans_motion2d_to_hoop_coord_sys(motion2d, hoop_bb)
            # motion2d = trans_motion2d_to_hips_coord_sys(motion2d)
            # Convert units
            motion2d_v2 = motion2d_v2 * np.array([scale_factor_x, scale_factor_y]).reshape((1, 2, 1))
            all_joints.append(motion2d_v2)

        all_joints = np.concatenate(all_joints, axis=2)
        meanpose = np.mean(all_joints, axis=2)
        stdpose = np.std(all_joints, axis=2)
        stdpose[np.where(stdpose == 0)] = 1e-9
        return meanpose, stdpose