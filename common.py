import os
# from functional import utils
from utils import utils
import torch
import numpy as np
import os.path as osp


class Config:
    name = None
    device = None
    dbg_mode = False
    objective_mode = None

    # dataset paths
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
    mot_en_channels = None
    body_en_channels = None
    cls_head_dims = None
    trj_head_dims = None
    # view_en_channels = None
    # de_channels = None

    # training settings
    # use_triplet = True
    # triplet_margin = 1
    # triplet_weight = 1
    # use_footvel_loss = False
    # foot_idx = [20, 21, 26, 27]
    # footvel_loss_weight = 0.1

    pre_release_n_frames = 35
    post_release_n_frames = 10

    poly_deg = 2

    nr_epochs = 250
    # batch_size = 24
    num_workers = 0  # TODO
    lr = 1e-3
    train_set_len = 323
    val_set_len = 50
    extras_set_len = 528
    save_frequency = 20
    # val_frequency = 14  # 10

    def initialize(self, args):
        self.name = args.name if hasattr(args, 'name') else 'skeleton'
        self.dbg_mode = args.dbg_mode if hasattr(args, 'dbg_mode') else False
        self.exp_dir = osp.join(self.save_dir, 'exp_' + self.name)
        self.log_dir = osp.join(self.exp_dir, 'log/')
        self.model_dir = osp.join(self.exp_dir, 'model/')
        utils.ensure_dirs([self.log_dir, self.model_dir])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.objective_mode = args.obj_mode if hasattr(args, 'obj_mode') else 'cls'
        self.nr_epochs = args.n_epochs if hasattr(args, 'n_epochs') else 250
        # self.poly_deg = args.poly_deg if hasattr(args, 'poly_deg') else 2
        self.meanpose_path = './bbfts_data/meanpose.npy'
        self.stdpose_path = './bbfts_data/stdpose.npy'

        if self.objective_mode == 'cls':
            self.mot_en_channels = [self.len_joints + 2, 64, 96, 128]
            self.body_en_channels = [self.len_joints + 2, 32, 48, 64]
            self.cls_head_dims = [768, 192, 48, 2]

        elif self.objective_mode == 'trj':
            self.mot_en_channels = [self.len_joints + 2, 64, 96, 128]
            self.body_en_channels = [self.len_joints + 2, 32, 48, 64]
            self.cls_head_dims = [768, 192, 48, 2]
            self.trj_head_dims = [768, 192, 48, 4]

            # self.meanpose_path = './bbfts_data/extras/meanpose.npy'
            # self.stdpose_path = './bbfts_data/extras/stdpose.npy'
        else:
            raise ValueError('Bad objective mode, must be trj (trajectory) or cls (classification)')


config = Config()