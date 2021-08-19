# from utils.operators import openpose2motion
from scipy.ndimage import gaussian_filter1d
import json
import os
import os.path as osp
import numpy as np
import argparse
import cv2
import math
import torch
import torchvision
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms
from PIL import Image
import pandas as pd
# from utils.utils import ensure_dir


JOINTS_DIR = 'joints'
CLIPS_DIR = 'clips'
MOTION_DIR = 'motion'
# BBALL_BB_DIR = 'bball_trajectories'
# SHOT_TRAJECTORY_DIR = 'shot_trajectory'
IOU_THRESHOLD = 0.3


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, help='fpath to dataset dir')
    parser.add_argument('--out-file', type=str, default='bball_trajectory.csv', help='out .csv file')
    parser.add_argument('--checkpoint', type=str, help='fpath to mask rcnn model weights')
    parser.add_argument('--num-samples', type=int, default=7, help='Num of frames to sample for ft shooter detection')
    parser.add_argument('--w-smooth', action='store_true', default=False, help='whether to smooth motion')

    args = parser.parse_args()
    return args


def get_instance_segmentation_model(num_classes=2):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


# def is_ball_in_hands_by_idx_n_eps(motion, time_i, bball_bb, eps=5):
#     """ Checking if the ball is in the shooters hands. We consider a positive response if
#     either one of his hands are in eps proximity to the ball's bounding box center. If not, we also
#     verify ball is located higher than the shooter, to avoid cases like dribbling.
#     """
#     in_hands = True
#     bball_center = np.array([0.25 * (bball_bb[0] + bball_bb[2]), 0.25 * (bball_bb[1] + bball_bb[3])])
#     r_hand_pos = motion[4, :, time_i]
#     l_hand_pos = motion[7, :, time_i]
#     r_hand_dist = np.linalg.norm(bball_center - r_hand_pos)
#     l_hand_dist = np.linalg.norm(bball_center - l_hand_pos)
#     if r_hand_dist > eps and l_hand_dist > eps:
#         head_pos_y = motion[0][1][time_i]
#         neck_pos_y = motion[1][1][time_i]
#         if (head_pos_y != 0 and head_pos_y >= bball_center[1]) or (neck_pos_y != 0 and neck_pos_y >= bball_center[1]):
#             # Checking the ball is located higher than the person (e.g not a dribble)
#             in_hands = False
#
#     return in_hands
#
#
# def is_ball_in_ascending_trajectory(a_bball_bb):
#     """ Checking if the ball is in an ascending trajectory by observing the
#     y coordinate of the ball's bounding box center across the given array coords"""
#     is_ascending = True
#     prev_bball_bb = a_bball_bb[0]
#     for i in range(1, len(a_bball_bb)):
#         curr_bball_bb = a_bball_bb[i]
#         if curr_bball_bb[1] > prev_bball_bb[1]:
#             # Comparing Y values
#             prev_bball_bb = curr_bball_bb
#         else:
#             is_ascending = False
#             break
#
#     return is_ascending
#
#
# def is_ball_in_descending_trajectory(a_bball_bb):
#     """ Checking if the ball is in a descending trajectory by observing the
#     y coordinate of the ball's bounding box center across the given array coords"""
#     is_descending = True
#     prev_bball_bb = a_bball_bb[0]
#     for i in range(1, len(a_bball_bb)):
#         curr_bball_bb = a_bball_bb[i]
#         if curr_bball_bb[1] < prev_bball_bb[1]:
#             # Comparing Y values
#             prev_bball_bb = curr_bball_bb
#         else:
#             is_descending = False
#             break
#
#     return is_descending
#
#
# def find_shot_frame_index(motion, a_bball_bb):
#     """ finds the shot release frame index by checking that:
#     1. The ball is not in the shooter hand by distance of atleast eps
#     2. The ball is in an ascending trajectory (to avoid catching a dribble or pose noise errors)
#     Returning the frame index (int)
#     """
#     n_traj_asc_samples = 5
#     _, _, num_frames = motion.shape
#     for frame_i in range(num_frames):
#         in_hands = is_ball_in_hands_by_idx_n_eps(motion, frame_i, a_bball_bb[frame_i])
#         if not in_hands:
#             is_ascending = is_ball_in_ascending_trajectory(a_bball_bb[frame_i:frame_i + n_traj_asc_samples])
#             if is_ascending:
#                 return frame_i


def sample_n_frames_from_vid(video_path, num_samples=5, num_frames=None):
    assert osp.exists(video_path)

    selected_frames = []
    capture = cv2.VideoCapture(video_path)

    start = 0
    end = num_frames
    if num_frames is None:
        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        end = round(0.6 * num_frames)

    capture.set(1, start)  # set the starting frame of the capture

    curr_frame = start
    chosen_frame_indices = [x for x in range(start, end - start, math.ceil((end - start) / num_samples))]
    assert len(chosen_frame_indices) == num_samples
    curr_chosen_idx = 0
    while curr_frame < end and curr_chosen_idx < num_samples:
        res, image = capture.read()

        if curr_frame in chosen_frame_indices:
            if image is None or res is False:
                # sometimes OpenCV reads None's during a video, in which case we want to just skip
                chosen_frame_indices[curr_chosen_idx] += 1
                # TODO make sure we dont exceed end frames
            else:
                selected_frames.append(image)
                curr_chosen_idx += 1

        curr_frame += 1

    capture.release()
    cv2.destroyAllWindows()

    assert len(selected_frames) == num_samples
    return selected_frames


def calc_iou(first, second, canvas_w, canvas_h):

    first_canvas = np.zeros((canvas_w, canvas_h))
    second_canvas = np.zeros((canvas_w, canvas_h))
    first_canvas[int(first[0]):int(first[2]), int(first[1]):int(first[3])] = 1
    second_canvas[int(second[0]):int(second[2]), int(second[1]):int(second[3])] = 1
    intersect = (first_canvas * second_canvas).sum(1).sum(0)
    union = (first_canvas + second_canvas).sum(1).sum(0)
    iou = (intersect + 0.001) / (union - intersect + 0.001)

    return iou


def locate_ft_shooter_in_clip(model, clip_fpath, num_samples, num_frames):
    frame_samples = sample_n_frames_from_vid(clip_fpath, num_samples, num_frames)
    # frame_samples is a list of numpy arrays, each represents a frame
    predictions = []
    disp_imgs = []
    model.eval()
    loader = transforms.Compose([transforms.ToTensor()])
    with torch.no_grad():
        for curr_frame in frame_samples:

            rgb_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            t_pil_image = loader(pil_image).float()
            t_pil_image = t_pil_image.cuda()

            prediction = model([t_pil_image])
            predictions.append(prediction)
            disp_imgs.append(pil_image)  # TODO maybe use it for testing visualization
    # print(len(predictions))
    # for k in range(len(predictions)):
    #     print(predictions[k][0]['scores'])
    # for k in range(len(predictions)):
    #     print(predictions[k][0]['boxes'])
    # print(predictions[0][0]['boxes'][0].cpu().numpy())
    # print(predictions[3][0]['boxes'][1].cpu().numpy())
    # print(predictions[1][0]['boxes'][2].cpu().numpy())
    # exit()

    w, h = disp_imgs[0].size
    box_bins = []
    for i in range(len(predictions)):
        if len(predictions[i][0]['boxes']) == 0:
            continue
        # if predictions[i][0]['scores'][0].item() < 0.5:
        #     continue
        curr_box = predictions[i][0]['boxes'][0].cpu().numpy()
        # Uncomment for case where 2nd top guy is FT shooter
        # if len(predictions[i][0]['boxes']) > 1:
        #     curr_box = predictions[i][0]['boxes'][1].cpu().numpy()
        if len(box_bins) == 0:
            box_bins.append([curr_box])
            continue

        found_bin = False
        for j in range(len(box_bins)):
            curr_iou = calc_iou(box_bins[j][0], curr_box, w, h)  # TODO maybe compare all not just 0
            if curr_iou > IOU_THRESHOLD:
                box_bins[j].append(curr_box)
                found_bin = True
                break

        if found_bin is False:
            box_bins.append([curr_box])
    if len(box_bins) == 0:
        return None

    final_box_i = box_bins.index(max(box_bins, key=len))
    min_x = sorted(box_bins[final_box_i], key=lambda x: x[0])[0][0] - 10
    max_x = sorted(box_bins[final_box_i], key=lambda x: x[2], reverse=True)[0][2] + 10
    min_y = sorted(box_bins[final_box_i], key=lambda x: x[1])[0][1] - 10
    max_y = sorted(box_bins[final_box_i], key=lambda x: x[3], reverse=True)[0][3] + 10

    final_box = [min_x, min_y, max_x, max_y]

    return final_box


def prepare_model(state_dict):
    # First we need to prepare model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # TODO: device = next(model.parameters()).device
    # our dataset has two classes only - background and person
    num_classes = 2

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)

    model.load_state_dict(torch.load(state_dict))
    model = model.cuda()  # TODO redundant?

    return model


def calc_two_poses_dist(pose1, pose2):
    dist = 0
    for joint_idx in range(len(pose1)):
        # Comparing non 0 valued joints
        if pose1[joint_idx][0] != 0 and pose2[joint_idx][0] != 0:
            dist += np.abs(pose1[joint_idx][0] - pose2[joint_idx][0]) + \
                                np.abs(pose1[joint_idx][1] - pose2[joint_idx][1])
    return dist


def find_closest_pose_numpy(np_poses_arr, target_pose):
    curr_min = 100000
    closest_pose = None
    for curr_person_joints in np_poses_arr:
        curr_person_dist = calc_two_poses_dist(curr_person_joints, target_pose)
        if curr_person_dist < curr_min:
            curr_min = curr_person_dist
            closest_pose = curr_person_joints

    # assert curr_min < 100000
    # assert closest_pose is not None
    return closest_pose


def find_closest_pose(poses_arr, target_pose):
    curr_min = 100000
    closest_pose = None
    for person_pose_info in poses_arr:
        curr_person_joints = np.array(person_pose_info['pose_keypoints_2d']).reshape((-1, 3))[:15, :2]
        curr_person_dist = calc_two_poses_dist(curr_person_joints, target_pose)
        if curr_person_dist < curr_min:
            curr_min = curr_person_dist
            closest_pose = curr_person_joints

    # assert curr_min < 100000
    # assert closest_pose is not None
    return closest_pose


def fill_zero_joints(joint_motion):
    # Handling first frame
    if joint_motion[0][0] == 0:
        for t in range(1, len(joint_motion)):
            if joint_motion[t][0] != 0:
                joint_motion[0][0] = joint_motion[t][0]
                joint_motion[0][1] = joint_motion[t][1]
                print('FIRST')
                break
    # assert joint_motion[0][0] != 0 and joint_motion[0][1] != 0
    if joint_motion[0][0] == 0 and joint_motion[0][1] == 0:
        return joint_motion.T

    # Handling last frame
    if joint_motion[len(joint_motion) - 1][0] == 0:
        for t in range(len(joint_motion) - 2, 0, -1):
            if joint_motion[t][0] != 0:
                joint_motion[len(joint_motion) - 1][0] = joint_motion[t][0]
                joint_motion[len(joint_motion) - 1][1] = joint_motion[t][1]
                print('Last')
                break

    # assert joint_motion[len(joint_motion) - 1][0] != 0 and joint_motion[len(joint_motion) - 1][1] != 0

    for i in range(len(joint_motion)):
        if joint_motion[i][0] == 0:
            last_good_joint = joint_motion[i - 1]
            next_good_joint = None
            zeros_count = 1
            for j in range(i + 1, len(joint_motion)):
                if joint_motion[j][0] == 0:
                    zeros_count += 1
                else:
                    next_good_joint = joint_motion[j]
                    break

            step_size_x = (next_good_joint[0] - last_good_joint[0]) / (zeros_count + 1)
            step_size_y = (next_good_joint[1] - last_good_joint[1]) / (zeros_count + 1)
            for k in range(zeros_count):
                assert joint_motion[i + k][0] == 0 and joint_motion[i + k][1] == 0
                joint_motion[i + k][0] = last_good_joint[0] + step_size_x * (k + 1)
                joint_motion[i + k][1] = last_good_joint[1] + step_size_y * (k + 1)

    for i in range(len(joint_motion)):
        if joint_motion[i][0] == 0:
            raise ValueError('!!!!!! STILL HAS 0 IN MOTION !!!!!!!!!!!!!!!!!!!!!!!')

    return joint_motion.T


def is_point_in_rectangle(rectangle, point, name=None):
    """ Returns True if the given point=(X,Y) is inside the rectangle """
    if point[0] != 0 and \
            rectangle[0] < point[0] < rectangle[2] and rectangle[1] < point[1] < rectangle[3]:
        if name is not None:
            print(name)
        return True
    return False


def num_points_in_rectangle(rectangle, points):
    """ Returns True if exists a point in the points list that is inside the rectangle """
    cnt_in = 0
    for point in points:
        if is_point_in_rectangle(rectangle, point):
            cnt_in += 1
    return cnt_in


def num_main_joints_in_box(bounding_box, joints):
    """
    Checking that Head, Right Arm, Left Arm, Hips, Right Upper Leg and Left Upper Leg are inside the bounding box
    and in the correct positing inside
    """
    head = joints[0]
    neck = joints[1]
    hips = joints[8]
    r_leg = joints[9]
    l_leg = joints[12]
    box_vert_len = np.abs(bounding_box[3] - bounding_box[1])
    box_horz_len = np.abs(bounding_box[2] - bounding_box[0])
    trimmed_box_max_y = bounding_box[1] + (box_vert_len * 0.35)
    trimmed_box_min_y = bounding_box[3] - (box_vert_len * 0.35)
    extended_box_min_x = bounding_box[0] - (box_horz_len * 0.18)
    extended_box_max_x = bounding_box[2] + (box_horz_len * 0.18)
    hips_based_box = [extended_box_min_x, trimmed_box_max_y, extended_box_max_x, bounding_box[3]]
    head_based_box = [extended_box_min_x, bounding_box[1], extended_box_max_x, trimmed_box_min_y]
    if num_points_in_rectangle(hips_based_box, [hips, l_leg, r_leg]) > 0 and \
            num_points_in_rectangle(head_based_box, [head, neck]) > 0:
        return True
    return False


def count_joints_in_box(bounding_box, joints):
    cnt_in = num_main_joints_in_box(bounding_box, joints)
    non_main_joints_indices = [2, 3, 4, 5, 6, 7, 10, 11, 13, 14]
    for i_joint in range(len(non_main_joints_indices)):
        if is_point_in_rectangle(bounding_box, joints[i_joint]):
            cnt_in += 1
    return cnt_in


def openpose2motionv2(json_dir, ft_bounding_box, scale=1.0, smooth=True):

    json_files = sorted(os.listdir(json_dir))
    json_files = [osp.join(json_dir, x) for x in json_files]

    motion = []
    for j, path in enumerate(json_files):
        with open(path) as f:
            jointDict = json.load(f)
            people_poses_arr = jointDict['people']
            joint_candidate = []
            for i, person_pose_info in enumerate(people_poses_arr):
                curr_joint = np.array(person_pose_info['pose_keypoints_2d']).reshape((-1, 3))[:15, :2]
                # TODO handle confidences?
                if num_main_joints_in_box(ft_bounding_box, curr_joint) > 0:
                    joint_candidate.append(curr_joint)

            selected_pose = None
            if len(joint_candidate) == 1:
                if len(motion) > 0:
                    # We need to verify we didn't get the wrong person because after shot release players move around
                    dist_from_prev = calc_two_poses_dist(motion[-1], joint_candidate[0])
                    if dist_from_prev > 400:
                        selected_pose = find_closest_pose(people_poses_arr, motion[-1])
                    else:
                        selected_pose = joint_candidate[0]
                    # print(f'Frame idx: {j}, Dist: {dist_from_prev}')
                else:
                    # First Frame case
                    selected_pose = joint_candidate[0]
                    # motion.append(joint_candidate[0])
            elif len(joint_candidate) == 0:
                if len(motion) > 0:
                    # Did not find any ft shooter but found in previous frame
                    selected_pose = find_closest_pose(people_poses_arr, motion[-1])
                    # motion.append(closest_person_joints)
                else:
                    print(f'!!!!!!!!! Did not find any ft shooters in frame {j} And Motion is Empty !!!!!!!!!')
            else:
                # More than 1 pose (main joints) found in ft shooter bounding box
                if len(motion) > 0:
                    selected_pose = find_closest_pose_numpy(joint_candidate, motion[-1])
                    # motion.append(closest_person_joints)
                else:
                    # First Frame case, but found multiple ft shooters - we now look at all of their joints
                    max_num_inside_joints = 0
                    # max_joints_pose = None
                    for curr_joint_candidate in joint_candidate:
                        curr_num_inside_joints = count_joints_in_box(ft_bounding_box, curr_joint_candidate)
                        if curr_num_inside_joints > max_num_inside_joints:
                            max_num_inside_joints = curr_num_inside_joints
                            selected_pose = curr_joint_candidate

            if selected_pose is not None:
                motion.append(selected_pose)
            else:
                print(f'!!!!!!!!! Selected pose is None in frame {j} !!!!!!!!!')


    max_dist = 0
    for i in range(len(motion) - 1):
        final_dist_from_prev = calc_two_poses_dist(motion[i], motion[i + 1])
        if final_dist_from_prev > max_dist:
            max_dist = final_dist_from_prev
    print(f'Motion Max Dist: {max_dist}')

    motion = np.stack(motion, axis=2)

    # Perform interpolation to remedy zeros
    # test_motion = motion.copy()

    n_joints, _, n_frames = motion.shape

    for i in range(n_joints):
        motion[i] = fill_zero_joints(motion[i].T)

    # j_dim, xy, n_fr = motion.shape
    # for t in range(n_fr):
    #     print('\n--------------------------------------------------------------')
    #     for j_idx in range(j_dim):
    #         if j_idx in [2,3,4,5,6,7]:
    #             print(f'({int(np.round(test_motion[j_idx, 0, t]))},{int(np.round(test_motion[j_idx, 1, t]))})', end=',')
    #     print()
    #     for j_idx in range(j_dim):
    #         if j_idx in [2, 3, 4, 5, 6, 7]:
    #             print(f'({int(np.round(motion[j_idx, 0, t]))},{int(np.round(motion[j_idx, 1, t]))})', end=',')

    if smooth:
        motion = gaussian_filter1d(motion, sigma=2, axis=-1)
    motion = motion * scale

    # zeros_cnt = 0
    # zeros_joints = []
    # for ii in range(len(motion[0][0])):
    #     for jj in range(len(motion[:, 0, ii])):
    #         if motion[jj][0][ii] == 0 and motion[jj][0][ii] == 0:
    #             zeros_cnt += 1
    #             zeros_joints.append(jj)
    # if zeros_cnt > 0:
    #     print(f'{zeros_cnt} + {zeros_joints}')
    # print(len(json_files))
    # a,b,c = motion.shape
    # print(motion.shape)
    # for i in range(c):
    #     for j in range(15):
    #         print(f'({motion[j][0][i]}, {motion[j][1][i]})', end=' ')
    #     print()
    return motion


def json2npy(data_dir, state_dict, num_samples, smooth):
    # preparing model
    model = prepare_model(state_dict)
    clips_dir_fpath = osp.join(data_dir, CLIPS_DIR)
    joints_dir_fpath = osp.join(data_dir, JOINTS_DIR)
    out_dir = osp.join(data_dir, MOTION_DIR)
    # ensure_dir(out_dir)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    vids_kp_dirs = os.listdir(joints_dir_fpath)
    for i, clip_name in enumerate(vids_kp_dirs):
        print(f'====== {i} - {clip_name} =====')
        # First we need to find the ft shooter in clip (bounding box)
        # if int(clip_name) not in [634]:
        #     continue
        curr_clip_fpath = osp.join(clips_dir_fpath, clip_name)
        curr_clip_fpath = f'{curr_clip_fpath}.mp4'
        ft_bounding_box = locate_ft_shooter_in_clip(model, curr_clip_fpath, num_samples=num_samples, num_frames=50)
        if ft_bounding_box is None:
            print('!!!!! Failed Detecting a FT Shooter !!!!!')
            continue
        # Second we extract all poses into a matrix
        clip_joints_dir_fpath = osp.join(joints_dir_fpath, clip_name)
        motion = openpose2motionv2(clip_joints_dir_fpath, ft_bounding_box, smooth=smooth)
        # returned motion shape is (J, 2, max_frame) and belongs to the free throws shooter
        # Here i am saving a matrix representing motion in 40 frames
        save_fpath = osp.join(out_dir, clip_name)
        save_fpath = f'{save_fpath}.npy'
        print(save_fpath)
        np.save(save_fpath, motion)


if __name__ == '__main__':
    args = parse_args()
    json2npy(args.data_dir, args.checkpoint, args.num_samples, args.w_smooth)
