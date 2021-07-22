from utils.operators import trans_motion2d_to_hips_coord_sys, trans_motion2d_to_hoop_coord_sys, calc_pixels_to_real_units_scaling_factor
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
        assert phase in ['train', 'test']
        assert config.labels_file.endswith('.csv')
        self.data_fpath = osp.join(config.data_dir, osp.join(phase, 'motion'))
        self.video_names = os.listdir(self.data_fpath)
        df = pd.read_csv(osp.join(config.data_dir, config.labels_file), header=0)
        self.labels_df = df.loc[df['phase'] == phase]
        self.phase = phase
        self.pre_rel_n_frames = config.pre_release_n_frames
        self.post_rel_n_frames = config.post_release_n_frames

        self.hoops_df = pd.read_csv(osp.join(config.data_dir, osp.join(phase, 'hoops_info.csv')), header=0)
        self.scale_factoring_map = {}
        mean_pose, std_pose = self.get_meanpose(config)

        self.aug = config.aug
        self.transforms = transforms.Compose([
            NormalizeMotion(mean_pose=mean_pose, std_pose=std_pose),
            Resize(scale=(config.nr_joints * 2, -1)),
            # GaussianNoise(mean_pose=0., std_pose=0.01),
            # RandomZeroMask(p=0.03),
            ToTensor()
        ])
        # self.cnt = 0
        # self.all_joints_glob = []

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

        if self.aug:
            hoop_bb = self.hoops_df.loc[self.hoops_df['name'] == f'{vid_name}.npy']['hoop'].item().split(',')
            if vid_name not in self.scale_factoring_map:
                # calculate scaling factor (from pixels to real world units, e.g Feet x alpha)
                scale_factor_x, scale_factor_y = calc_pixels_to_real_units_scaling_factor(motion, hoop_bb, alpha=0.4)
                self.scale_factoring_map[vid_name] = {'X': scale_factor_x, 'Y': scale_factor_y}
                # self.cnt += 1

            # Transform motion coordinate system
            motion = trans_motion2d_to_hoop_coord_sys(motion, hoop_bb)
            # motion = trans_motion2d_to_hips_coord_sys(motion)
            # Convert units
            motion = motion * np.array([self.scale_factoring_map[vid_name]['X'], self.scale_factoring_map[vid_name]['Y']]).reshape((1, 2, 1))

            # motion = (motion - self.mean_pose[:, :, np.newaxis]) / self.std_pose[:, :, np.newaxis]

            # if self.cnt <= 323:
            #     self.all_joints_glob.append(motion)
            motion = self.transforms(motion)
            # if self.cnt >= 323:
            #     all_joints_glob_np = np.concatenate(self.all_joints_glob, axis=2)
            #     meanpose = np.mean(all_joints_glob_np, axis=2, dtype=np.float64)
            #     stdpose = np.std(all_joints_glob_np, axis=2, dtype=np.float64)
            #     stdpose[np.where(stdpose == 0)] = 1e-9
            #     print('MeanPose after scaling')
            #     print(meanpose)
            #     print()
            #     print('StdPose after scaling')
            #     print(stdpose)
            #     exit()
            # motion = motion.reshape((30, -1))
            # motion = torch.Tensor(motion)

        sample = {'name': vid_name, 'motion': motion, 'label': label}

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

    # @staticmethod
    # def gen_aug_param(rotate=False):
    #     if rotate:
    #         return {'ratio': np.random.uniform(0.8, 1.2),
    #                 'roll': np.random.uniform((-np.pi / 9, -np.pi / 9, -np.pi / 6), (np.pi / 9, np.pi / 9, np.pi / 6))}
    #     else:
    #         return {'ratio': np.random.uniform(0.5, 1.5)}
    #
    # @staticmethod
    # def augmentation(dataset, param=None):
    #     """
    #     :param dataset: numpy array of size (joints, 3, len_frames)
    #     :return:
    #     """
    #     if param is None:
    #         return dataset, param
    #
    #     # rotate
    #     if 'roll' in param.keys():
    #         cx, cy, cz = np.cos(param['roll'])
    #         sx, sy, sz = np.sin(param['roll'])
    #         mat33_x = np.array([
    #             [1, 0, 0],
    #             [0, cx, -sx],
    #             [0, sx, cx]
    #         ], dtype='float')
    #         mat33_y = np.array([
    #             [cy, 0, sy],
    #             [0, 1, 0],
    #             [-sy, 0, cy]
    #         ], dtype='float')
    #         mat33_z = np.array([
    #             [cz, -sz, 0],
    #             [sz, cz, 0],
    #             [0, 0, 1]
    #         ], dtype='float')
    #         dataset = mat33_x @ mat33_y @ mat33_z @ dataset
    #
    #     # scale
    #     if 'ratio' in param.keys():
    #         dataset = dataset * param['ratio']
    #
    #     return dataset, param

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
            # motion2d = trans_motion2d_to_hips_coord_sys(motion2d)
            motion2d = trans_motion2d_to_hoop_coord_sys(motion2d, hoop_bb)
            # Convert units
            motion2d = motion2d * np.array([scale_factor_x, scale_factor_y]).reshape((1, 2, 1))
            all_joints.append(motion2d)

        all_joints = np.concatenate(all_joints, axis=2)
        meanpose = np.mean(all_joints, axis=2)
        stdpose = np.std(all_joints, axis=2)
        stdpose[np.where(stdpose == 0)] = 1e-9
        return meanpose, stdpose