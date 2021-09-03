import numpy as np
import torch
import os
import os.path as osp
import json
from scipy.ndimage import gaussian_filter1d

# Joint Names:  Head, Neck,
#               RightArm, RightForeArm, RightHand,
#               LeftArm, LeftForeArm, LeftHand,
#               Hips,
#               RightUpLeg, RightLeg, RightFoot,
#               LeftUpLeg, LeftLeg, LeftFoot


def calc_polynomial_coeff_by_points_n_deg(x, y, deg=2):
    assert len(x) == len(y)
    return np.polyfit(x, y, deg)


def trans_motion2d_to_hips_coord_sys(motion2d):
    # subtract centers to local coordinates
    centers = motion2d[8, :, :]
    motion_proj = motion2d - centers

    # adding velocity
    velocity = np.c_[np.zeros((2, 1)), centers[:, 1:] - centers[:, :-1]].reshape(1, 2, -1)
    motion_proj = np.r_[motion_proj[:8], motion_proj[9:], velocity]

    # motion_proj = np.r_[motion_proj[:8], centers.reshape((1, 2, -1)), motion_proj[9:]]

    return motion_proj


def trans_motion2d_to_hoop_coord_sys(motion, hoop_pos):
    _, _, num_frames = motion.shape
    hoop_bb_x = int(hoop_pos[2])
    hoop_bb_y = (int(hoop_pos[1]) + int(hoop_pos[3])) // 2
    hoop_np = np.array([hoop_bb_x, hoop_bb_y])

    motion = -1 * (motion - np.tile(hoop_np, (num_frames, 1)).T)
    # motion[:, 1, :] *= -1 TODO this version reduces training acc to 0.9 but hurts val by about 0.2-0.4 acc also
    return motion


def inv_trans_motion2d_to_hoop_coord_sys(motion, hoop_pos):
    _, _, num_frames = motion.shape
    hoop_bb_x = int(hoop_pos[2])
    hoop_bb_y = (int(hoop_pos[1]) + int(hoop_pos[3])) // 2
    hoop_np = np.array([hoop_bb_x, hoop_bb_y])

    motion = -1 * motion + np.tile(hoop_np, (num_frames, 1)).T
    return motion


def inv_trans_motion2d_to_hoop_coord_sys_w_scales(scaled_motion, hoop_pos, scale_factor_x, scale_factor_y):
    scaled_motion[:, 0, :] /= scale_factor_x
    scaled_motion[:, 1, :] /= scale_factor_y
    return inv_trans_motion2d_to_hoop_coord_sys(scaled_motion, hoop_pos)


def calc_pixels_to_real_units_scaling_factor(motion, hoop_pos, alpha=0.4):
    real_hoop_to_floor_dist = 10.0 * alpha
    real_backboard_to_ft_line_dist = 15.0 * alpha

    # We assume Hoop is on Right side of the frame (Even if motion was flipped !)
    assert int(hoop_pos[2]) > int(hoop_pos[0])
    hoop_bb_x = int(hoop_pos[2])
    hoop_bb_y = (int(hoop_pos[1]) + int(hoop_pos[3])) // 2
    person_y = -1
    person_x = -1
    hips_x = motion[8, 0, -10:-5]
    hips_x = hips_x[hips_x != 0]
    hips_y = motion[8, 1, -10:-5]
    hips_y = hips_y[hips_y != 0]
    head_y = motion[0, 1, -10:-5]
    head_y = head_y[head_y != 0]

    r_leg_x = motion[11, 0, -10:-5]
    r_leg_x = r_leg_x[r_leg_x != 0]
    r_leg_y = motion[11, 1, -10:-5]
    r_leg_y = r_leg_y[r_leg_y != 0]

    l_leg_x = motion[14, 0, -10:-5]
    l_leg_x = l_leg_x[l_leg_x != 0]
    l_leg_y = motion[14, 1, -10:-5]
    l_leg_y = l_leg_y[l_leg_y != 0]
    # Get X and Y coordinate of free throw line center
    if len(l_leg_y) > 0 and len(r_leg_y) > 0:
        person_y = (np.mean(l_leg_y) + np.mean(r_leg_y)) // 2
        person_x = max(np.mean(l_leg_x), np.mean(r_leg_x)) + 2
    else:
        # Feet was not found by pose detection, probably occluded, so we calculate it
        print('!!!!!! Did not find any feet to Scale motion !!!!')
        person_y = np.mean(hips_y) + abs(np.mean(head_y) - np.mean(hips_y))
        person_x = np.mean(hips_x) + 2

    scale_factor_y = float(real_hoop_to_floor_dist / abs(hoop_bb_y - person_y))
    scale_factor_x = float(real_backboard_to_ft_line_dist / abs(hoop_bb_x - person_x))

    return scale_factor_x, scale_factor_y


def trans_motion_inv(motion, sx=256, sy=256, velocity=None):
    if velocity is None:
        velocity = motion[-1].copy()
    motion_inv = np.r_[motion[:8], np.zeros((1, 2, motion.shape[-1])), motion[8:-1]]

    # restore centre position
    centers = np.zeros_like(velocity)
    sum = 0
    for i in range(motion.shape[-1]):
        sum += velocity[:, i]
        centers[:, i] = sum
    centers += np.array([[sx], [sy]])

    return motion_inv + centers.reshape((1, 2, -1))


def normalize_motion_inv(motion, mean_pose, std_pose):
    if len(motion.shape) == 2:
        motion = motion.reshape(-1, 2, motion.shape[-1])
    return motion * std_pose[:, :, np.newaxis] + mean_pose[:, :, np.newaxis]


def postprocess_motion2d(motion, mean_pose, std_pose, sx=256, sy=256):
    motion = motion.detach().cpu().numpy()[0].reshape(-1, 2, motion.shape[-1])
    motion = trans_motion_inv(normalize_motion_inv(motion, mean_pose, std_pose), sx, sy)
    return motion


def openpose2motion(json_dir, scale=1.0, smooth=True, max_frame=None):
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


def get_foot_vel(batch_motion, foot_idx):
    return batch_motion[:, foot_idx, 1:] - batch_motion[:, foot_idx, :-1] + batch_motion[:, -2:, 1:].repeat(1, 2, 1)
