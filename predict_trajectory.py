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
from utils.utils import cycle


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default='bbfts_data', help='path to data dir of a certain phase')
    parser.add_argument('--subset', type=str, default='test', help='path to data dir of a certain phase')
    parser.add_argument('--motion-dir', type=str, default='motion', help='name of joints dir')
    parser.add_argument('--clips-dir', type=str, default='clips', help='name of clips dir')
    parser.add_argument('--out-dir', type=str, default='visualizations/test', help='full path to output dir')

    parser.add_argument('--shot-traj-dir', type=str, default='shot_trajectories', help='name of shot trajectory dir')
    parser.add_argument('--labels-info', type=str, default='bbfts_data/bbfts_labels.csv', help='full path to labels file')

    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False, help="specify gpu ids")
    parser.add_argument('--checkpoint', type=str, default=None, help='path to weights')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    data_dir = osp.join(args.data_dir, args.subset)
    args.obj_mode = 'trj'
    shot_rel_df = pd.read_csv(args.labels_info, header=0)

    utils.ensure_dir(args.out_dir)
    shot_traj_dir = osp.join(data_dir, args.shot_traj_dir)

    motion_dir = osp.join(data_dir, args.motion_dir)

    clips_dir = osp.join(data_dir, args.clips_dir)
    clips = [x for x in os.listdir(clips_dir) if x.endswith('.mp4')]
    hoops_df = pd.read_csv(osp.join(data_dir, 'hoops_info.csv'), header=0)

    color = utils.hex2rgb('#a50b69#b73b87#db9dc3')

    config.initialize(args)
    net = get_network(config)
    net = net.to(config.device)
    tr_moder = get_training_moderator(config, net, lr=1)
    tr_moder.pre_train_load_network(args.checkpoint)
    data_len = len(clips)
    data_loader = get_dataloader(args.subset, config, data_len, config.num_workers, shuffle=False)
    data_loader = cycle(data_loader)
    data = next(data_loader)

    outputs, losses = tr_moder.val_func(data)
    trj_output = outputs['trj'].detach().cpu().numpy()

    for data_i in range(data_len):
        curr_vid_name = data['name'][data_i].item()

        save_path = osp.join(args.out_dir, f'{curr_vid_name}.png')

        curr_shot_rl_frame = int(shot_rel_df.loc[shot_rel_df['video_name'] == int(curr_vid_name)]['shot_frame'].item())

        curr_motion_name = f'{curr_vid_name}.npy'
        curr_hoop_bb = hoops_df.loc[hoops_df['name'] == curr_motion_name]['hoop'].item().split(',')

        shot_traj = np.load(osp.join(shot_traj_dir, curr_motion_name))
        shot_pose = np.load(osp.join(motion_dir, curr_motion_name))[:, :, curr_shot_rl_frame]

        scale_factor_x , scale_factor_y = data['scale'][data_i].numpy()

        # lbl_first_ball_coords = shot_traj[1, :]
        lbl_high_ball_coords = shot_traj[1, :]
        lbl_last_ball_coords = shot_traj[-1, :]

        for i in range(2, len(shot_traj) - 2):
            if shot_traj[i][1] < lbl_high_ball_coords[1]:
                lbl_high_ball_coords = shot_traj[i]

        trj_labels = [lbl_high_ball_coords,
                      lbl_last_ball_coords]
        # trj_labels = [lbl_first_ball_coords,
        #               lbl_high_ball_coords,
        #               lbl_last_ball_coords]

        # first_x, first_y, high_x, high_y, last_x, last_y = trj_output[data_i]
        high_x, high_y, last_x, last_y = trj_output[data_i]

        # Below is relative to first shot
        # high_x_in_pixels_diff = float(high_x / scale_factor_x)
        # high_y_in_pixels_diff = float(high_y / scale_factor_y)
        # high_global_x = high_x_in_pixels_diff + shot_traj[0][0]
        # high_global_y = shot_traj[0][1] - high_y_in_pixels_diff
        #
        # last_x_in_pixels_diff = float(last_x / scale_factor_x)
        # last_y_in_pixels_diff = float(last_y / scale_factor_y)
        # last_global_x = last_x_in_pixels_diff + shot_traj[0][0]
        # last_global_y = shot_traj[0][1] - last_y_in_pixels_diff

        # Below is relative to hoop
        # first_x_in_pixels_diff = float(first_x / scale_factor_x)
        # first_y_in_pixels_diff = float(first_y / scale_factor_y)
        # first_global_x = int(curr_hoop_bb[2]) - first_x_in_pixels_diff
        # first_global_y = ((int(curr_hoop_bb[3]) + int(curr_hoop_bb[1])) / 2) - first_y_in_pixels_diff

        high_x_in_pixels_diff = float(high_x / scale_factor_x)
        high_y_in_pixels_diff = float(high_y / scale_factor_y)
        high_global_x = int(curr_hoop_bb[2]) - high_x_in_pixels_diff
        high_global_y = ((int(curr_hoop_bb[3]) + int(curr_hoop_bb[1])) / 2) - high_y_in_pixels_diff

        last_x_in_pixels_diff = float(last_x / scale_factor_x)
        last_y_in_pixels_diff = float(last_y / scale_factor_y)
        last_global_x = int(curr_hoop_bb[2]) - last_x_in_pixels_diff
        last_global_y = ((int(curr_hoop_bb[3]) + int(curr_hoop_bb[1])) / 2) - last_y_in_pixels_diff

        # predicted_trj = [np.array([first_global_x, first_global_y]),
        #                  np.array([high_global_x, high_global_y]),
        #                  np.array([last_global_x, last_global_y])]
        predicted_trj = [np.array([high_global_x, high_global_y]),
                         np.array([last_global_x, last_global_y])]
        print(f'Prediction:   {predicted_trj}')
        print(f'Ground Truth: {trj_labels}')
        make_shot_trajectory_image(shot_pose, h=720, w=1280, save_path=save_path, colors=color,
                                   shot_traj_gt=trj_labels, shot_traj=predicted_trj,
                                   hoop_bb=curr_hoop_bb)

        print(f'====== Finished {data_i} - {curr_vid_name} ======')
