from utils.operators import normalize_motion, trans_motion2d, preprocess_motion2d
from torch.utils.data import Dataset
import os
import os.path as osp
import numpy as np
import torch
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
        self.meanpose_path = config.meanpose_path
        self.stdpose_path = config.stdpose_path
        self.aug = config.aug

        self.mean_pose, self.std_pose = get_meanpose(config)

    def __len__(self):
        return len(os.listdir(self.data_fpath))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            print('IS TENSOR IN GETITEM')
            idx = idx.tolist()  # TODO
        man_fix = 35
        man_fix_top = 10
        vid_name = self.labels_df.iloc[idx]['video_name']
        # Create one-hot label vector
        label_idx = self.labels_df.iloc[idx]['label']
        label = torch.from_numpy(np.array([label_idx])).type(torch.long)

        vid_fpath = osp.join(self.data_fpath, f'{vid_name}.npy')
        motion = np.load(vid_fpath)
        shot_frame = int(self.labels_df.iloc[idx]['shot_frame'])
        assert shot_frame != ''

        # if vid_name == 170:
        #     shot_frame = 51

        if shot_frame <= man_fix:
            n_diff = man_fix - shot_frame
            shot_frame += n_diff
            first_pose = np.copy(motion[:, :, 0])
            first_pose = first_pose[..., np.newaxis]
            first_pose_matrix = np.repeat(first_pose, n_diff, axis=2)
            motion = np.concatenate((first_pose_matrix, motion), axis=2)

        motion = motion[:, :, shot_frame - man_fix:shot_frame + man_fix_top]
        # _, _, t = motion.shape
        # print(f'{vid_name} - {shot_frame} = {t}')
        motion = self.preprocessing(motion)

        sample = {'name': vid_name, 'motion': motion, 'label': label}
        return sample

    def preprocessing(self, motion):
        # Here i need to normalize 2d joints matrix
        # joints_arr = np.load(item)
        # if self.aug:
        #     TODO HERE
        #     motion3d, param = self.augmentation(motion3d, param)
        motion = normalize_motion(motion, self.mean_pose, self.std_pose)
        motion = motion.reshape((-1, motion.shape[-1]))  # Should be (joints*2, len_frames)
        motion = torch.Tensor(motion)  # TODO from_numpy ?
        # motion = preprocess_motion2d(motion, self.mean_pose, self.std_pose)

        return motion

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


def get_meanpose(config):
    meanpose_path = config.meanpose_path
    stdpose_path = config.stdpose_path
    if osp.exists(meanpose_path) and osp.exists(stdpose_path):
        meanpose = np.load(meanpose_path)
        stdpose = np.load(stdpose_path)
    else:
        meanpose, stdpose = gen_meanpose(config)
        np.save(meanpose_path, meanpose)
        np.save(stdpose_path, stdpose)
        print("meanpose saved at {}".format(meanpose_path))
        print("stdpose saved at {}".format(stdpose_path))
    return meanpose, stdpose


def gen_meanpose(config):
    all_paths = glob.glob(osp.join(config.data_dir, 'train', 'motion/*.npy'))  # TODO this should be the paths to all the clips_joints.npy
    all_joints = []

    for path in all_paths:
        motion2d = np.load(path)
        motion2d = motion2d[:, :, :45]
        motion2d_preprocessed = trans_motion2d(motion2d)
        all_joints.append(motion2d_preprocessed)
        # local3d = None
        # if config.view_angles is None:
        #     motion_proj = trans_motion3d(motion3d, local3d)
        #     all_joints.append(motion_proj)
        # else:
        #     for angle in config.view_angles:
        #         local3d = get_local3d(motion3d, angle)
        #         motion_proj = trans_motion3d(motion3d.copy(), local3d)
        #         all_joints.append(motion_proj)

    all_joints = np.concatenate(all_joints, axis=2)
    meanpose = np.mean(all_joints, axis=2)
    stdpose = np.std(all_joints, axis=2)
    stdpose[np.where(stdpose == 0)] = 1e-9
    return meanpose, stdpose