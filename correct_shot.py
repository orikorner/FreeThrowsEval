from dataset import get_dataloader
from common import config
from model import get_network
from utils.visualization import motion2video, hex2rgb, joints2image_w_colors
from utils.operators import inv_trans_motion2d_to_hoop_coord_sys_w_scales
import torch
import argparse
import torch.nn as nn
import os
import os.path as osp
import pandas as pd
from utils.utils import cycle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import cv2
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
import imageio
from tqdm import tqdm
from PIL import Image
from matplotlib import cm

N_JOINTS = 15
N_FRAMES = 45


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='bbfts_data', help='path to data dir')
    parser.add_argument('--motion-dir', type=str, default='motion', help='name of joints dir')
    parser.add_argument('--subset', type=str, default='test', help='test/train/extras')
    parser.add_argument('--vid-name', type=str, default=None, help='single video name, for single mode')
    parser.add_argument('--out-name', type=str, default=None, help='full path to output file')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to weights')
    parser.add_argument('--n-epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--wmax', type=int, default=0.85, help='weight given to maximum')
    parser.add_argument('--mode', type=str, default='gradmap', help='either gradmap or motvid')
    # parser.add_argument('-c', '--continue', dest='continue_path', type=str, required=False)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False, help="specify gpu ids")

    args = parser.parse_args()
    return args


def motion_grad_intensity_to_dislay_color(levels, value):
    if value <= levels[1]:
        return 71, 71, 184
    elif levels[1] < value <= levels[2]:
        return 0, 255, 255
    elif levels[2] < value <= levels[3]:
        return 0, 255, 0
    elif levels[3] < value <= levels[4]:
        return 255, 128, 0
    else:
        return 102, 0, 0


def get_motion_grad_map(vid_name, labels_info_df, data, net, criterion, wmax=0.85, n_epochs=1, lr=0.1):
    # np.set_printoptions(precision=3)
    colored_motion = np.zeros((N_JOINTS, N_FRAMES, 3))
    wmin = 1 - wmax

    item_idx = (data['name'] == int(vid_name)).nonzero(as_tuple=False).item()
    # name = data['name'][item_idx]
    # scale_factor_x, scale_factor_y = data['scale'][item_idx].numpy()
    cls_labels = data['cls_labels'][item_idx].to(config.device).reshape(-1)

    data_motion = data['motion'][item_idx].unsqueeze(0).to(config.device)
    data_motion.requires_grad_(True)

    optimizer = torch.optim.Adam([data_motion], lr=lr)
    data_motion_grad = None
    levels = None
    for e in range(n_epochs):
        optimizer.zero_grad()
        outputs, _ = net(data_motion)

        _, preds_tensor = torch.max(outputs, 1)
        # preds = np.squeeze(preds_tensor.cpu().clone().numpy())
        # print(f'{preds}: {F.softmax(outputs[0], dim=0)[preds].item()}')
        loss = criterion(outputs, cls_labels)
        loss.backward()

        data_motion_grad = data_motion.grad.squeeze(0).detach().cpu().numpy()
        data_motion_grad = gaussian_filter1d(data_motion_grad, sigma=2.5, axis=1, mode='nearest')
        data_motion_grad = np.array(list(map(lambda x: abs(x), data_motion_grad)))

        levels = MaxNLocator(nbins=5).tick_values(0, data_motion_grad.max())

        for t_idx in range(N_FRAMES):
            for j_idx in range(N_JOINTS):
                max_val = max(data_motion_grad[j_idx * 2, t_idx], data_motion_grad[j_idx * 2 + 1, t_idx])
                min_val = min(data_motion_grad[j_idx * 2, t_idx], data_motion_grad[j_idx * 2 + 1, t_idx])
                joint_sp_tmp_val = wmax * max_val + wmin * min_val

                colored_motion[j_idx, t_idx] = motion_grad_intensity_to_dislay_color(levels, joint_sp_tmp_val)

                data_motion_grad[j_idx * 2, t_idx] = joint_sp_tmp_val
                data_motion_grad[j_idx * 2 + 1, t_idx] = joint_sp_tmp_val

        # optimizer.step()

    # Below is case for viewing motion after optimizer steps, but havent been tested after modifications...
    # updated_motion = data_motion.squeeze(0).detach().cpu().numpy().reshape((15, 2, -1))
    #
    # updated_motion = updated_motion * std_pose[:, :, np.newaxis] + mean_pose[:, :, np.newaxis]
    #
    # updated_motion = inv_trans_motion2d_to_hoop_coord_sys_w_scales(updated_motion.copy(), hoop_bb,
    #                                                                scale_factor_x, scale_factor_y)

    # motion2video(updated_motion, h=720, w=1280, save_path=f'{args.out_name}.mp4',
    #              colors=hex2rgb('#a50b69#b73b87#db9dc3'),
    #              shot_rel_frame=shot_rl_frame, shot_traj=None, hoop_bbs=None,
    #              transparency=False, motion_tgt=None, fps=25, save_frame=False)

    return data_motion_grad, colored_motion, levels


def main():
    args = parse_args()

    # mean_pose = np.load(osp.join(args.data_dir, 'meanpose.npy'))
    # std_pose = np.load(osp.join(args.data_dir, 'stdpose.npy'))

    data_dir = osp.join(args.data_dir, args.subset)
    motion_dir = osp.join(data_dir, args.motion_dir)

    args.obj_mode = 'trj'
    config.initialize(args)

    labels_info_df = pd.read_csv(osp.join(config.data_dir, config.labels_file), header=0)
    labels_info_df = labels_info_df.loc[labels_info_df['phase'] == args.subset]

    # hoops_df = pd.read_csv(osp.join(data_dir, 'hoops_info.csv'), header=0)

    net = get_network(config)
    net = net.to(config.device)
    net.load_state_dict(torch.load(args.checkpoint))
    net.eval()
    criterion = nn.CrossEntropyLoss()
    data_len = len(os.listdir(motion_dir))
    data_loader = get_dataloader(args.subset, config, data_len, config.num_workers, shuffle=False)
    data = next(cycle(data_loader))
    data_motion_grad = np.zeros((30, 45))
    colored_motion = None
    if args.vid_name is not None:
        data_motion_grad, colored_motion, levels = get_motion_grad_map(args.vid_name, labels_info_df,
                                                                       data, net, criterion,
                                                                       args.wmax, args.n_epochs)
    else:
        for data_name in data['name']:
            curr_data_motion_grad, _, _ = get_motion_grad_map(data_name.item(), labels_info_df,
                                                              data, net, criterion,
                                                              args.wmax, args.n_epochs)
            data_motion_grad += curr_data_motion_grad

        data_motion_grad /= data_len

    if args.mode == 'gradmap':
        # Matplotlib version
        x = np.arange(0, 45, 1)
        y = np.array(['Head_X', 'Head_Y', 'Neck_X', 'Neck_Y',
                      'RightArm_X', 'RightArm_Y', 'RightForeArm_X', 'RightForeArm_Y', 'RightHand_X', 'RightHand_Y',
                      'LeftArm_X', 'LeftArm_Y', 'LeftForeArm_X', 'LeftForeArm_Y', 'LeftHand_X', 'LeftHand_Y',
                      'Hips_X', 'Hips_Y',
                      'RightUpLeg_X', 'RightUpLeg_Y', 'RightLeg_X', 'RightLeg_Y', 'RightFoot_X', 'RightFoot_Y',
                      'LeftUpLeg_X', 'LeftUpLeg_Y', 'LeftLeg_X', 'LeftLeg_Y', 'LeftFoot_X', 'LeftFoot_Y'])

        # levels = MaxNLocator(nbins=5).tick_values(data_motion_grad.min(), data_motion_grad.max())
        levels = MaxNLocator(nbins=5).tick_values(0, data_motion_grad.max())
        cmap = plt.get_cmap('jet')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        fig, ax = plt.subplots(figsize=(30, 30))
        im = ax.pcolormesh(x, y, data_motion_grad, shading='auto', cmap=cmap, norm=norm)
        fig.colorbar(im, ax=ax)
        if args.vid_name is not None:
            ax.set_title(f'Video: {args.vid_name}')
        else:
            ax.set_title(f'Video: {args.subset}')

        if args.out_name is not None:
            plt.savefig(f'{args.out_name}.png')

        plt.show()

    elif args.mode == 'motvid' and args.vid_name is not None:
        true_motion = np.load(osp.join(motion_dir, f'{args.vid_name}.npy'))
        shot_rl_frame = int(labels_info_df.loc[labels_info_df['video_name'] == int(args.vid_name)]['shot_frame'].item())
        # hoop_bb = hoops_df.loc[hoops_df['name'] == f'{args.vid_name}.npy']['hoop'].item().split(',')
        videowriter = imageio.get_writer(f'{args.out_name}.mp4', fps=25)

        shot_released = False
        for i in tqdm(range(45)):
            if shot_rl_frame is not None and i >= shot_rl_frame:
                shot_released = True
            img = joints2image_w_colors(joints_position=true_motion[:, :, i],
                                        colors_info=colored_motion[:, i],
                                        limb_color=hex2rgb('#a50b69#b73b87#db9dc3'),
                                        shot_released=shot_released)
            videowriter.append_data(img)
        videowriter.close()


if __name__ == '__main__':
    main()
