import os
# from functional import utils
from utils import general
import torch
import numpy as np
import os.path as osp


class Config:
    name = None
    device = None
    aug = True
    # dataset paths
    # data_dir = './mixamo_data'
    data_dir = './bbfts_data'
    labels_file = 'bbfts_labels.csv'
    meanpose_path = None
    stdpose_path = None

    # training paths
    save_dir = './train_log'
    exp_dir = None
    log_dir = None
    model_dir = None

    # dataset info
    # TODO
    img_size = (512, 512)
    # unit = 128
    nr_joints = 15
    len_joints = 2 * nr_joints - 2
    # view_angles = [(0, 0, -np.pi / 2),
    #                (0, 0, -np.pi / 3),
    #                (0, 0, -np.pi / 6),
    #                (0, 0, 0),
    #                (0, 0, np.pi / 6),
    #                (0, 0, np.pi / 3),
    #                (0, 0, np.pi / 2)]

    # network channels
    # mot_en_channels = None
    body_en_channels = None
    # view_en_channels = None
    # de_channels = None

    # training settings
    # use_triplet = True
    # triplet_margin = 1
    # triplet_weight = 1
    # use_footvel_loss = False
    # foot_idx = [20, 21, 26, 27]
    # footvel_loss_weight = 0.1

    nr_epochs = 300
    batch_size = 64
    num_workers = 0 # TODO
    lr = 1e-3

    save_frequency = 50
    val_frequency = 100
    # visualize_frequency = 500

    def initialize(self, args):
        self.name = args.name if hasattr(args, 'name') else 'tmp'
        # self.name = args.name if hasattr(args, 'name') else 'skeleton'
        # self.name = args.name if hasattr(args, 'name') else 'full'
        # self.use_triplet = not args.disable_triplet if hasattr(args, 'disable_triplet') else None
        # self.use_footvel_loss = args.use_footvel_loss if hasattr(args, 'use_footvel_loss') else None

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.aug = args.aug
        self.exp_dir = osp.join(self.save_dir, 'exp_' + self.name)
        self.log_dir = osp.join(self.exp_dir, 'log/')
        self.model_dir = osp.join(self.exp_dir, 'model/')
        general.ensure_dirs([self.log_dir, self.model_dir])

        # if self.name == 'skeleton':
        if True:
            # self.mot_en_channels = [self.len_joints + 2, 64, 96, 128]
            self.body_en_channels = [self.len_joints, 32, 48, 64]
            # self.de_channels = [self.mot_en_channels[-1] + self.body_en_channels[-1], 128, 64, self.len_joints + 2]
            # self.view_angles = None

            # self.meanpose_path = './mixamo_data/meanpose.npy'
            self.meanpose_path = './bbfts_data/meanpose.npy'
            # self.stdpose_path = './mixamo_data/stdpose.npy'
            self.stdpose_path = './bbfts_data/stdpose.npy'
        # elif self.name == 'view':
        #     self.mot_en_channels = [self.len_joints + 2, 64, 96, 128]
        #     self.view_en_channels = [self.len_joints, 64, 96, 128, 32]
        #     self.de_channels = [self.mot_en_channels[-1] + self.view_en_channels[-1], 128, 64, self.len_joints + 2]
        #
        #     self.meanpose_path = './mixamo_data/meanpose_with_view.npy'
        #     self.stdpose_path = './mixamo_data/stdpose_with_view.npy'
        # else:
        #     self.mot_en_channels = [self.len_joints + 2, 64, 96, 128]
        #     self.body_en_channels = [self.len_joints, 32, 48, 64, 16]
        #     self.view_en_channels = [self.len_joints, 32, 48, 64, 8]
        #     self.de_channels = [self.mot_en_channels[-1] + self.body_en_channels[-1] + self.view_en_channels[-1],
        #                         128, 64, self.len_joints + 2]
        #
        #     self.meanpose_path = './mixamo_data/meanpose_with_view.npy'
        #     self.stdpose_path = './mixamo_data/stdpose_with_view.npy'


config = Config()