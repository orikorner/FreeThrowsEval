# from utils.operators import openpose2motion
from scipy.ndimage import gaussian_filter1d
import json
import os
import os.path as osp
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, help='fpath to dataset dir')
    parser.add_argument('--out-dir', type=str, help='fpath to dataset dir')

    args = parser.parse_args()
    return args


def openpose2motionv2(json_dir, scale=1.0, smooth=True, max_frame=None):
    json_files = sorted(os.listdir(json_dir))
    length = max_frame if max_frame is not None else len(json_files) // 8 * 8
    json_files = json_files[:length]
    json_files = [osp.join(json_dir, x) for x in json_files]

    motion = []
    for path in json_files:
        with open(path) as f:
            jointDict = json.load(f)
            joint = np.array(jointDict['people'][0]['pose_keypoints_2d']).reshape((-1, 3))[:15, :2]
            if len(motion) > 0:
                joint[np.where(joint == 0)] = motion[-1][np.where(joint == 0)]
            motion.append(joint)

    for i in range(len(motion) - 1, 0, -1):
        motion[i - 1][np.where(motion[i - 1] == 0)] = motion[i][np.where(motion[i - 1] == 0)]

    motion = np.stack(motion, axis=2)

    if smooth:
        motion = gaussian_filter1d(motion, sigma=2, axis=-1)
    motion = motion * scale
    return motion


def json2npy(data_dir, out_dir):
    vids_kp_dirs = os.listdir(data_dir)
    for vid_joints_dir in vids_kp_dirs:
        vid_joints_dir_fpath = osp.join(data_dir, vid_joints_dir)
        joints_json_files = os.listdir(vid_joints_dir_fpath)
        num_frames = len(joints_json_files) # TODO
        motion = openpose2motionv2(vid_joints_dir_fpath, max_frame=42, smooth=False)
        # returned motion shape is (J, 2 max_frame)
        # Here i am saving a matrix representing motion in 42 frames
        # TODO need to find actually person first!
        save_fpath = osp.join(out_dir, vid_joints_dir)
        save_fpath = f'{save_fpath}.npy'
        print(save_fpath)
        np.save(save_fpath, motion)


if __name__ == '__main__':
    args = parse_args()
    json2npy(args.data_dir, args.out_dir)