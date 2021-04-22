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


JOINTS_DIR = 'joints'
CLIPS_DIR = 'clips'
MOTION_DIR = 'motion'
IOU_THRESHOLD = 0.3


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, help='fpath to dataset dir')
    parser.add_argument('--checkpoint', type=str, help='fpath to mask rcnn model weights')
    parser.add_argument('--num-samples', type=int, default=5, help='Num of frames to sample for ft shooter detection')

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

    #thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    #return thresholded  # Or thresholded.mean()
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
            # image = Variable(image, requires_grad=False)
            t_pil_image = t_pil_image.cuda()

            prediction = model([t_pil_image])
            predictions.append(prediction)
            disp_imgs.append(pil_image)  # TODO maybe use it for testing visualization

    w, h = disp_imgs[0].size
    box_bins = []
    if len(predictions[0][0]['boxes']) > 0:
        box_bins = [[predictions[0][0]['boxes'][0].cpu().numpy()]]
    for i in range(1, len(predictions)):
        if len(predictions[i][0]['boxes']) == 0:
            continue
        curr_box = predictions[i][0]['boxes'][0].cpu().numpy()

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

    # construct an optimizer
    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005,
    #                             momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,
    #                                                gamma=0.1)

    model.load_state_dict(torch.load(state_dict))
    model = model.cuda()  # TODO redundant?

    return model


def is_point_in_rectangle(rectangle, point, name=None):
    if rectangle[0] < point[0] < rectangle[2] and rectangle[1] < point[1] < rectangle[3]:
        if name is not None:
            print(name)
        return True
    return False


def count_main_joints_in_box(bounding_box, joints):
    parts = [joints[0], joints[2], joints[5], joints[8], joints[9], joints[12]]  # head, r arm, l arm, hips, ru leg, lu leg
    cnt_in = 0
    for part in parts:
        if is_point_in_rectangle(bounding_box, part):
            cnt_in += 1
    return cnt_in


def openpose2motionv2(json_dir, ft_bounding_box, scale=1.0, smooth=True, max_frame=None):
    length = max_frame - 1
    json_files = sorted(os.listdir(json_dir))
    json_files = [osp.join(json_dir, x) for x in json_files]

    box_vert_len = np.abs(ft_bounding_box[3] - ft_bounding_box[1])
    box_horz_len = np.abs(ft_bounding_box[2] - ft_bounding_box[0])
    trimmed_box_max_y = ft_bounding_box[1] + (box_vert_len * 0.35)
    trimmed_box_min_y = ft_bounding_box[3] - (box_vert_len * 0.35)
    extended_box_min_x = ft_bounding_box[0] - (box_horz_len * 0.18)
    extended_box_max_x = ft_bounding_box[2] + (box_horz_len * 0.18)
    hips_based_box = [extended_box_min_x, trimmed_box_max_y, extended_box_max_x, ft_bounding_box[3]]
    head_based_box = [extended_box_min_x, ft_bounding_box[1], extended_box_max_x, trimmed_box_min_y]
    box_center = (0.25 * (ft_bounding_box[0] + ft_bounding_box[2]), 0.25 * (ft_bounding_box[1] + ft_bounding_box[3]))
    motion = []
    for j, path in enumerate(json_files):
        with open(path) as f:
            jointDict = json.load(f)
            people_poses_arr = jointDict['people']
            joint_candidate = []
            for i, person_pose_info in enumerate(people_poses_arr):
                curr_joint = np.array(person_pose_info['pose_keypoints_2d']).reshape((-1, 3))[:15, :2]
                top_check_pass = False
                bottom_check_pass = False
                curr_hips = curr_joint[8]
                curr_r_leg = curr_joint[9]
                curr_l_leg = curr_joint[12]
                curr_head = curr_joint[0]
                curr_neck = curr_joint[1]
                if curr_hips[0] != 0 and is_point_in_rectangle(hips_based_box, curr_hips) or \
                        curr_l_leg[0] != 0 and is_point_in_rectangle(hips_based_box, curr_l_leg) or \
                        curr_r_leg[0] != 0 and is_point_in_rectangle(hips_based_box, curr_r_leg):
                    bottom_check_pass = True

                if bottom_check_pass:
                    if curr_head[0] != 0 and is_point_in_rectangle(head_based_box, curr_head) or \
                            curr_neck[0] != 0 and is_point_in_rectangle(head_based_box, curr_neck):
                        top_check_pass = True

                if top_check_pass and bottom_check_pass:
                    if len(motion) > 0:
                        # fills joints with 0 value to be the same as their previous value (in previous frame)
                        curr_joint[np.where(curr_joint == 0)] = motion[-1][np.where(curr_joint == 0)]
                    joint_candidate.append(curr_joint)

            if len(joint_candidate) > 0:
                if len(joint_candidate) > 1:
                    print(f'Longer than 1 pose detected in frame: {j}')
                    if len(motion) == 0:
                        joint_candidate.sort(
                            key=lambda p: np.abs(p[8][0] - box_center[0]) + np.abs(p[8][1] - box_center[1]) +
                                          np.abs(p[0][0] - box_center[0]) + np.abs(p[0][1] - box_center[1]))
                    else:
                        last_hips = np.array(motion[-1][8])
                        last_head = motion[-1][0]
                        joint_candidate.sort(
                            key=lambda p: np.abs(p[8][0] - last_hips[0]) + np.abs(p[8][1] - last_hips[1]) +
                                          np.abs(p[0][0] - last_head[0]) + np.abs(p[0][1] - last_head[1]))

                motion.append(joint_candidate[0])
            else:
                length += 1

        if j >= length:
            break

    if len(motion) != max_frame:
        print(f'!!!!!!! len_mot: {len(motion)} - mf: {max_frame}')
    
    for i in range(len(motion) - 1, 0, -1):
        motion[i - 1][np.where(motion[i - 1] == 0)] = motion[i][np.where(motion[i - 1] == 0)]

    motion = np.stack(motion, axis=2)

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
    return motion


def json2npy(data_dir, state_dict, num_samples):
    # preparing model
    model = prepare_model(state_dict)
    clips_dir_fpath = osp.join(data_dir, CLIPS_DIR)
    joints_dir_fpath = osp.join(data_dir, JOINTS_DIR)
    out_dir = osp.join(data_dir, MOTION_DIR)
    vids_kp_dirs = os.listdir(joints_dir_fpath)
    for i, clip_name in enumerate(vids_kp_dirs):
        print(f'====== {i} - {clip_name} =====')
        # First we need to find the ft shooter in clip (bounding box)
        curr_clip_fpath = osp.join(clips_dir_fpath, clip_name)
        curr_clip_fpath = f'{curr_clip_fpath}.mp4'
        ft_bounding_box = locate_ft_shooter_in_clip(model, curr_clip_fpath, num_samples=num_samples, num_frames=50)
        if ft_bounding_box is None:
            print('!!!!! Failed Detecting a FT Shooter !!!!!')#123,113,104,195
            continue
        # Second we extract all poses into a matrix
        clip_joints_dir_fpath = osp.join(joints_dir_fpath, clip_name)
        joints_json_files = os.listdir(clip_joints_dir_fpath)
        num_frames = min(len(joints_json_files), 40)  # TODO
        assert num_frames == 40
        # num_frames = len(joints_json_files)
        motion = openpose2motionv2(clip_joints_dir_fpath, ft_bounding_box, max_frame=num_frames, smooth=False)
        # returned motion shape is (J, 2, max_frame) and belongs to the free throws shooter
        # Here i am saving a matrix representing motion in 42 frames
        save_fpath = osp.join(out_dir, clip_name)
        save_fpath = f'{save_fpath}.npy'
        print(save_fpath)
        np.save(save_fpath, motion)


if __name__ == '__main__':
    args = parse_args()
    json2npy(args.data_dir, args.checkpoint, args.num_samples)