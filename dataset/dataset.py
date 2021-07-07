from utils.operators import trans_motion2d # normalize_motion, trans_motion2d, preprocess_motion2d
from .augmentations import NormalizeMotion, Resize, ToTensor, GaussianNoise
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
        mean_pose, std_pose = self.get_meanpose(config)
        self.aug = config.aug
        self.transforms = transforms.Compose([
            NormalizeMotion(mean_pose=mean_pose, std_pose=std_pose),
            Resize(scale=(config.nr_joints * 2, -1)),
            # GaussianNoise(mean_pose=0., std_pose=1.5),
            ToTensor()
        ])

    def __len__(self):
        return len(os.listdir(self.data_fpath))

    def norm_mot_ori(self, motion, hoop_x, hoop_y):
        # print(motion.shape)
        person_y = -1
        person_x = -1
        hips = motion[8, :, 0]
        head = motion[0, :, 0]
        neck = motion[1, :, 0]

        div = 0
        if neck[0] != 0:
            person_x += neck[0]
            div += 1
        if head[0] != 0:
            person_x += head[0]
            div += 1
        if hips[0] != 0:
            person_x += hips[0]
            div += 1
        person_x /= div

        r_leg = motion[11, :, 0]
        l_leg = motion[14, :, 0]
        if r_leg[1] != 0 and l_leg[1] != 0:
            person_y = (r_leg[1] + l_leg[1]) // 2
        else:
            person_y = hips[1] - abs(head[1] - hips[1])

        person_hoop_dist = np.sqrt(((hoop_y - person_y)**2 + (hoop_x - person_x)**2))
        pixel_units = float(15.0 / person_hoop_dist)  # 1 d in pixels = 1 units

        # TODO convert person - hoop dist to pix to pix dist for joints and use it to normalize!

    def __getitem__(self, idx):
        vid_name = self.labels_df.iloc[idx]['video_name']
        # Create one-hot label vector
        label_idx = self.labels_df.iloc[idx]['label']
        label = torch.from_numpy(np.array([label_idx])).type(torch.long)

        vid_fpath = osp.join(self.data_fpath, f'{vid_name}.npy')
        motion = np.load(vid_fpath)
        shot_frame = int(self.labels_df.iloc[idx]['shot_frame'])

        motion, shot_frame = self.duplicate_pose_by_shot_frame(motion, shot_frame)
        motion = motion[:, :, shot_frame - self.pre_rel_n_frames:shot_frame + self.post_rel_n_frames]
        if self.aug:
            # hoop_bb = self.hoops_df.loc[self.hoops_df['name'] == f'{vid_name}.npy']['hoop'].item().split(',')
            # hoop_bb_x = int(hoop_bb[2])
            # hoop_bb_y = (int(hoop_bb[1]) + int(hoop_bb[3])) // 2
            # norm_mot_ori(motion)
            motion = self.transforms(motion)

        sample = {'name': vid_name, 'motion': motion, 'label': label}

        return sample

    def duplicate_pose_by_shot_frame(self, motion, shot_frame):
        if shot_frame <= self.pre_rel_n_frames:
            n_diff = self.pre_rel_n_frames - shot_frame
            shot_frame += n_diff
            first_pose = np.copy(motion[:, :, 0])
            first_pose = first_pose[..., np.newaxis]
            first_pose_matrix = np.repeat(first_pose, n_diff, axis=2)
            motion = np.concatenate((first_pose_matrix, motion), axis=2)

        return motion, shot_frame

    # def preprocessing(self, motion):
        # Here i need to normalize 2d joints matrix
        # joints_arr = np.load(item)
        # if self.aug:
        #     TODO HERE
        #     motion3d, param = self.augmentation(motion3d, param)

        # TODO below 3 were used
        #motion = normalize_motion(motion, self.mean_pose, self.std_pose)
        #motion = motion.reshape((-1, motion.shape[-1]))  # Should be (joints*2, len_frames)
        #motion = torch.Tensor(motion)  # TODO from_numpy ?

        # motion = preprocess_motion2d(motion, self.mean_pose, self.std_pose)

        # return motion

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
        all_paths = sorted(glob.glob(osp.join(config.data_dir, 'train', 'motion/*.npy')))   # TODO this should be the paths to all the clips_joints.npy
        all_joints = []
        for path in all_paths:
            motion2d = np.load(path)
            curr_vid_name = osp.splitext(osp.basename(path))[0]
            curr_shot_frame = int(self.labels_df.loc[self.labels_df['video_name'] == int(curr_vid_name)]['shot_frame'].item())
            motion2d, curr_shot_frame = self.duplicate_pose_by_shot_frame(motion2d, curr_shot_frame)
            motion2d = motion2d[:, :, curr_shot_frame - self.pre_rel_n_frames:curr_shot_frame + self.post_rel_n_frames]
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