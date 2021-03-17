from utils.operators import openpose2motion
import os
import os.path as osp
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-dir', type=str, help='fpath to dataset dir')
    parser.add_argument('--out-dir', type=str, help='fpath to dataset dir')

    args = parser.parse_args()
    return args


def json2npy(data_dir, out_dir):
    vids_kp_dirs = os.listdir(data_dir)
    for vid_joints_dir in vids_kp_dirs:
        vid_joints_dir_fpath = osp.join(data_dir, vid_joints_dir)
        joints_json_files = os.listdir(vid_joints_dir_fpath)
        num_frames = len(joints_json_files) # TODO
        motion = openpose2motion(vid_joints_dir_fpath, max_frame=42)
        # Here i am saving a matrix representing motion in 42 frames
        # TODO need to find actually person first!
        save_fpath = osp.join(out_dir, vid_joints_dir)
        save_fpath = f'{save_fpath}.npy'
        print(save_fpath)
        np.save(save_fpath, motion)


if __name__ == '__main__':
    args = parse_args()
    json2npy(args.data_dir, args.out_dir)