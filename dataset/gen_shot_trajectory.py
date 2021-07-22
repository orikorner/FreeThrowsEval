import argparse
import os
import os.path as osp
import numpy as np
import pandas as pd

# from utils.utils import ensure_dir

MOTION_DIR = 'motion'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, help='fpath to dataset dir')
    parser.add_argument('--phase', type=str, default='train', help='test/train')
    parser.add_argument('--labels-file', type=str, default='bbfts_labels.csv', help='out .csv file')
    parser.add_argument('--shot-traj-dir', type=str, default='shot_trajectories', help='name of shot trajectory dir')
    parser.add_argument('--p-det-dir', type=str, default='processed_yolo_detections', help='dir of processed detections')

    args = parser.parse_args()
    return args


def get_bounding_box_center(bball_bb):
    bball_center = np.array([0.5 * (bball_bb[0] + bball_bb[2]), 0.5 * (bball_bb[1] + bball_bb[3])])
    return bball_center


def is_ball_in_hands_by_idx_n_eps(motion, time_i, bball_bb, eps=30):
    """ Checking if the ball is in the shooters hands. We consider a positive response if
    either one of his hands are in eps proximity to the ball's bounding box center. If not, we also
    verify ball is located higher than the shooter, to avoid cases like dribbling.
    """
    in_hands = True

    bball_center = get_bounding_box_center(bball_bb)
    if bball_center[0] == 0 and bball_center[1] == 0:
        return True

    r_hand_pos = motion[4, :, time_i]
    l_hand_pos = motion[7, :, time_i]
    r_hand_dist = np.linalg.norm(bball_center - r_hand_pos)
    l_hand_dist = np.linalg.norm(bball_center - l_hand_pos)
    if r_hand_dist > eps and l_hand_dist > eps:
        head_pos = motion[0, :, time_i]
        neck_pos = motion[1, :, time_i]
        if (head_pos[1] != 0 and head_pos[1] >= bball_center[1]) or (neck_pos[1] != 0 and neck_pos[1] >= bball_center[1]):
            # Checking the ball is located higher than the person (e.g not a dribble)
            if head_pos[0] == 0 or (head_pos[0] != 0 and abs(head_pos[0] - bball_center[0]) > 20):
                in_hands = False

    if not in_hands:
        print(f'{time_i}: ({r_hand_pos[0]},{r_hand_pos[1]}) {r_hand_dist} ({l_hand_pos[0]},{l_hand_pos[1]}) {l_hand_dist} ({bball_center[0]},{bball_center[1]}) {in_hands}')
    return in_hands


def is_ball_in_ascending_trajectory(a_bball_bb, hoop_center, n_traj_samples=4):
    """ Checking if the ball is in an ascending trajectory by observing the
    y coordinate of the ball's bounding box center across the given array coords.
    Also Checking that the ball is moving towards the hoop by comparing X coordinates of the ball and hoop.
    """
    is_ascending = True
    prev_bball_bb = get_bounding_box_center(a_bball_bb[0])
    prev_ball_hoop_dist = abs(hoop_center[0] - prev_bball_bb[0])
    for i in range(1, len(a_bball_bb)):
        if n_traj_samples == 0:
            break
        curr_bball_bb = get_bounding_box_center(a_bball_bb[i])
        if curr_bball_bb[0] == 0 and curr_bball_bb[1] == 0:
            continue
        curr_ball_hoop_dist = abs(hoop_center[0] - curr_bball_bb[0])
        if curr_bball_bb[1] < prev_bball_bb[1] and curr_ball_hoop_dist < prev_ball_hoop_dist:
            # Comparing Y values (Ascending) and Distance between ball and hoop (Shooting)
            prev_bball_bb = curr_bball_bb
            prev_ball_hoop_dist = curr_ball_hoop_dist
            n_traj_samples -= 1
        else:
            is_ascending = False
            break

    return is_ascending


def is_ball_in_descending_trajectory(a_bball_bb,  n_traj_samples=3):
    """ Checking if the ball is in a descending trajectory by observing the
    y coordinate of the ball's bounding box center across the given array coords"""
    is_descending = True
    prev_bball_bb = get_bounding_box_center(a_bball_bb[0])
    for i in range(1, len(a_bball_bb)):
        if n_traj_samples == 0:
            break
        curr_bball_bb = get_bounding_box_center(a_bball_bb[i])
        if curr_bball_bb[0] == 0 and curr_bball_bb[1] == 0:
            continue
        if curr_bball_bb[1] > prev_bball_bb[1]:
            # Comparing Y values
            prev_bball_bb = curr_bball_bb
            n_traj_samples -= 1
        else:
            is_descending = False
            break

    return is_descending


def find_hoop_in_array_by_offset(a_hoop_bb, offset=0):
    found_hoop_bb = None
    found = False
    for frame_i in range(offset, len(a_hoop_bb)):
        curr_hoop_bb = a_hoop_bb[frame_i]
        curr_hoop_center = get_bounding_box_center(curr_hoop_bb)
        if curr_hoop_center[0] != 0 or curr_hoop_center[1] != 0:
            found_hoop_bb = curr_hoop_bb
            found = True
            break

    if found_hoop_bb is None and offset > 0:
        # make another attempt..take any hoop we find
        for frame_i in range(offset - 1, -1, -1):
            curr_hoop_bb = a_hoop_bb[frame_i]
            curr_hoop_center = get_bounding_box_center(curr_hoop_bb)
            if curr_hoop_center[0] != 0 or curr_hoop_center[1] != 0:
                found_hoop_bb = curr_hoop_bb
                found = True
                break

    assert found_hoop_bb is not None
    return found_hoop_bb, found


def find_shot_frame_index(motion, a_bball_bb, a_hoop_bb):
    """ finds the shot release frame index by checking that:
    1. The ball is not in the shooter hand by distance higher than eps
    2. The ball is in an ascending trajectory (to avoid catching a dribble or pose noise errors)
    3. The ball is moving towards the hoop
    Returning the frame index (int)
    """
    approx_hoop_center = get_bounding_box_center(a_hoop_bb[0])
    if approx_hoop_center[0] == 0 and approx_hoop_center[1] == 0:
        for hoop_i in range(1, len(a_hoop_bb)):
            approx_hoop_center = get_bounding_box_center(a_hoop_bb[hoop_i])
            if approx_hoop_center[0] != 0 or approx_hoop_center[1] != 0:
                break

    for frame_i in range(len(a_bball_bb)):
        in_hands = is_ball_in_hands_by_idx_n_eps(motion, frame_i, a_bball_bb[frame_i])
        if not in_hands:
            is_ascending = is_ball_in_ascending_trajectory(a_bball_bb[frame_i:], approx_hoop_center)
            if is_ascending:
                return frame_i


def convert_ball_n_hoop_info_into_np(objs_info_fpath):
    """ Takes a single hoop and ball locations info file and parses is into a numpy array.
    every line in the file represents a frame, and every line is of format:
    obj_id x1 y1 x2 y2,obj_id x1 y1 x2 y2,
    we return 2 numpy arrays (ball and hoop) with the coordinates separated by a comma
    """
    bball_bb_info = []
    hoop_bb_info = []
    with open(objs_info_fpath, 'r') as objs_info_fp:
        for i, line in enumerate(objs_info_fp):
            if not line.strip():
                # Check if ball not found (empty line)
                bball_bb_info.append([0, 0, 0, 0])
                hoop_bb_info.append([0, 0, 0, 0])
                print(f'!!!!!!! DID NOT FIND ANY BASKETBALL in frame {i} !!!!!!! ')
            else:
                obj1, obj2 = line.split(',')
                obj1 = [int(x) for x in obj1.split(' ')]
                obj2 = [int(x) for x in obj2.split(' ')]
                # Object id coordinate
                obj1_id = obj1.pop(0)
                obj2_id = obj2.pop(0)
                if obj1_id == 1 and obj2_id == 0:
                    bball_bb_info.append(obj2)
                    hoop_bb_info.append(obj1)
                else:
                    bball_bb_info.append(obj1)
                    hoop_bb_info.append(obj2)

    # TODO here we need to handle missing balls
    return np.array(bball_bb_info), np.array(hoop_bb_info)


def make_ball_trajectory_array(a_bball_bb, a_hoop_bb, shot_frame_i=0, sample_interval=1, eps=10):
    """
    Samples the basketball's position in intervals. We start at shot frame and finish when
    the ball hits the hoop. We return the 'Hit' frame and the Ball Trajectory.
    """
    sampled_shot_trajectory = []
    ball_hit_hoop_frame = -1
    shot_dir = None
    prev_bball_center = None
    curr_hoop_bb = None
    started_descending = False
    for j in range(shot_frame_i, len(a_bball_bb), sample_interval):
        curr_bball_center = get_bounding_box_center(a_bball_bb[j])
        if curr_bball_center[0] == 0 and curr_bball_center[1] == 0:
            # Ball Not found
            continue

        curr_hoop_bb, found_hoop = find_hoop_in_array_by_offset(a_hoop_bb, offset=j)
        if shot_dir is None and found_hoop:
            if curr_hoop_bb[2] < curr_bball_center[0]:
                shot_dir = 'left'
            elif curr_hoop_bb[0] > curr_bball_center[0]:
                shot_dir = 'right'
            else:
                raise ValueError('!!! Wrong calculation of shot direction')

        if not started_descending:
            started_descending = is_ball_in_descending_trajectory(a_bball_bb[j:])

        # TODO perhaps check that ball hasn't moved too much, to help verify we have the correct ball
        if (shot_dir == 'left' and curr_hoop_bb[2] >= curr_bball_center[0]) or \
                (shot_dir == 'right' and curr_hoop_bb[0] <= curr_bball_center[0]) or \
                (started_descending and shot_dir == 'left' and curr_bball_center[0] > prev_bball_center[0]) or \
                (started_descending and shot_dir == 'right' and curr_bball_center[0] < prev_bball_center[0]):
            ball_hit_hoop_frame = j - 1
            break

        sampled_shot_trajectory.append(curr_bball_center)
        prev_bball_center = curr_bball_center

    if ball_hit_hoop_frame == -1:
        if (started_descending and shot_dir == 'left' and (prev_bball_center[0] - curr_hoop_bb[2]) < eps) or \
                (started_descending and shot_dir == 'right' and (curr_hoop_bb[0] - prev_bball_center[0]) < eps):
            ball_hit_hoop_frame = len(a_bball_bb) - 1
        else:
            raise ValueError('Ball did not hit Hoop Error')
    return np.array(sampled_shot_trajectory), ball_hit_hoop_frame


def create_shots_frames_labels(data_dir, phase, labels_file, shot_traj_dir, p_det_dir):
    """ Main Function - Finds the shot frame index using motion matrix and bounding box trajectories:
     Expects a p_det_dir containing txt files, each holds the ball position throughout the clip,
     i.e every line in each files represents the frame index, and every line is of format:
     label_id x1 y1 x2 y2
     """
    labels_fpath = osp.join(data_dir, labels_file)
    data_dir = osp.join(data_dir, phase)
    shot_traj_dir = osp.join(data_dir, shot_traj_dir)

    if not osp.exists(shot_traj_dir):
        os.makedirs(shot_traj_dir)

    motions_dir = osp.join(data_dir, MOTION_DIR)
    p_det_dir_fpath = osp.join(data_dir, p_det_dir)
    l_motion_file_names = os.listdir(motions_dir)

    shot_traj_dict = {}
    for i, curr_motion_fname in enumerate(l_motion_file_names):
        print(f'====== {i} - {curr_motion_fname} =====')

        # if curr_motion_fname not in ['109.npy']:
        #     continue
        # Getting FT Shooter's motion
        curr_motion = np.load(osp.join(motions_dir, curr_motion_fname))
        # Getting Ball and Hoop Bounding boxes locations as matrix
        curr_motion_fname = osp.splitext(curr_motion_fname)[0]

        curr_objs_info_fpath = f'{curr_motion_fname}.txt'
        curr_objs_info_fpath = osp.join(p_det_dir_fpath, curr_objs_info_fpath)
        a_bball_bb, a_hoop_bb = convert_ball_n_hoop_info_into_np(curr_objs_info_fpath)
        # Finding shot release frame
        shot_frame_i = find_shot_frame_index(curr_motion, a_bball_bb, a_hoop_bb)
        shot_traj_dict[curr_motion_fname] = shot_frame_i
        # Making shot trajectory
        # if curr_motion_fname == '109':
        #     continue
        a_shot_trajectory, ball_hit_hoop_frame = make_ball_trajectory_array(a_bball_bb, a_hoop_bb,
                                                                            shot_frame_i=shot_frame_i)
        print(f'Shot Release: {shot_frame_i} - Ball hit Hoop: {ball_hit_hoop_frame} - Trajectory length: {len(a_shot_trajectory)}')
        save_fpath = osp.join(shot_traj_dir, curr_motion_fname)
        save_fpath = f'{save_fpath}.npy'
        np.save(save_fpath, a_shot_trajectory)

    labels_df = pd.read_csv(labels_fpath, header=0)
    if 'shot_frame' not in labels_df.columns:
        labels_df['shot_frame'] = ""

    for k_vid_name, v_shot_frame in shot_traj_dict.items():
        labels_df.loc[labels_df.video_name == int(k_vid_name), 'shot_frame'] = int(v_shot_frame)

    labels_df.to_csv(labels_fpath, index=False)


if __name__ == '__main__':
    args = parse_args()

    create_shots_frames_labels(args.data_dir, args.phase, args.labels_file, args.shot_traj_dir, args.p_det_dir)
