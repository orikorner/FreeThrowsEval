import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import math
import argparse
from tqdm import tqdm
import utils
from PIL import Image
import imageio
import os
import os.path as osp
import pandas as pd
# from utils import hex2rgb
from common import config
from model import get_network
from dataset import get_dataloader
from moderator import get_training_moderator
from utils.visualization import make_shot_trajectory_image
from utils.operators import calc_polynomial_coeff_by_points_n_deg


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default='bbfts_data/train', help='path to data dir of a certain phase')
    parser.add_argument('--motion-dir', type=str, default='motion', help='name of joints dir')
    parser.add_argument('--clips-dir', type=str, default='clips', help='name of clips dir')
    parser.add_argument('--out-dir', type=str, default='visualizations/train', help='full path to output dir')

    parser.add_argument('--shot-traj-dir', type=str, default='shot_trajectories', help='name of shot trajectory dir')
    parser.add_argument('--labels-info', type=str, default='bbfts_data/bbfts_labels.csv', help='full path to labels file')

    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False, help="specify gpu ids")

    parser.add_argument('--checkpoint', type=str, default=None, help='path to weights')
    parser.add_argument('--poly-deg', type=int, default=2, help="ball trajectory polynomial degree")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.obj_mode = 'trj'
    shot_rel_df = pd.read_csv(args.labels_info, header=0)

    utils.ensure_dir(args.out_dir)
    shot_traj_dir = osp.join(args.data_dir, args.shot_traj_dir)

    motion_dir = osp.join(args.data_dir, args.motion_dir)
    motions = os.listdir(motion_dir)
    clips_dir = osp.join(args.data_dir, args.clips_dir)
    clips = [x for x in os.listdir(clips_dir) if x.endswith('.mp4')]
    hoops_df = pd.read_csv(osp.join(args.data_dir, 'hoops_info.csv'), header=0)

    assert len(motions) == len(clips)

    color1 = utils.hex2rgb('#a50b69#b73b87#db9dc3')

    config.initialize(args)
    net = get_network(config)
    net = net.to(config.device)
    tr_moder = get_training_moderator(config, net, lr=1)
    tr_moder.pre_train_load_network(args.checkpoint)

    val_loader = get_dataloader('test', config, config.val_set_len, config.num_workers, shuffle=False)
    dataiter = iter(val_loader)
    data = next(dataiter)

    outputs, losses = tr_moder.val_func(data)
    outputs = outputs.detach().cpu().numpy()

    for data_i in range(config.val_set_len):
        curr_vid_name = data['name'][data_i].item()

        save_path = osp.join(args.out_dir, f'{curr_vid_name}_deg{args.poly_deg}.png')
        curr_shot_rl_frame = int(shot_rel_df.loc[shot_rel_df['video_name'] == int(curr_vid_name)]['shot_frame'].item())
        curr_motion_name = f'{curr_vid_name}.npy'
        curr_hoop_bb = hoops_df.loc[hoops_df['name'] == curr_motion_name]['hoop'].item().split(',')

        shot_traj = np.load(osp.join(shot_traj_dir, curr_motion_name))
        shot_pose = np.load(osp.join(motion_dir, curr_motion_name))[:, :, curr_shot_rl_frame]

        scale_factor_x , scale_factor_y = data['scale'][data_i].numpy()
        poly_result_shot_traj = []
        if config.poly_deg == 2:
            a, b, c = outputs[data_i]
            for traj_i in range(len(shot_traj)):
                real_x = shot_traj[traj_i][0]
                # real_y = shot_traj[traj_i][1]
                x_in_traj_coords = real_x - shot_traj[0][0]
                scaled_x_in_traj_coords = x_in_traj_coords * scale_factor_x
                scaled_y_in_traj_coords = (a * (scaled_x_in_traj_coords ** 2)) + (b * scaled_x_in_traj_coords) + c
                y_in_traj_coords = scaled_y_in_traj_coords / scale_factor_y
                predicted_real_y = shot_traj[0][1] - y_in_traj_coords
                # print([int(real_x), int(predicted_real_y)])
                poly_result_shot_traj.append([int(real_x), int(predicted_real_y)])
        elif config.poly_deg == 3:
            a, b, c, d = outputs[data_i]
            for traj_i in range(len(shot_traj)):
                real_x = shot_traj[traj_i][0]
                # real_y = shot_traj[traj_i][1]
                x_in_traj_coords = real_x - shot_traj[0][0]
                scaled_x_in_traj_coords = x_in_traj_coords * scale_factor_x
                scaled_y_in_traj_coords = (a * (scaled_x_in_traj_coords ** 3)) + (b * (scaled_x_in_traj_coords ** 2)) + (c * scaled_x_in_traj_coords) + d
                y_in_traj_coords = scaled_y_in_traj_coords / scale_factor_y
                predicted_real_y = shot_traj[0][1] - y_in_traj_coords
                # print([int(real_x), int(predicted_real_y)])
                poly_result_shot_traj.append([int(real_x), int(predicted_real_y)])
        else:
            raise ValueError('Polynomial degree of shot trajectory must be 2 or 3')

        make_shot_trajectory_image(shot_pose, h=720, w=1280, save_path=save_path,
                                   colors=color1, shot_traj_gt=shot_traj, shot_traj=poly_result_shot_traj,
                                   hoop_bb=curr_hoop_bb)

        print(f'====== Finished Batch {data_i} ======')