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
    first_canvas[first[0]:first[2], first[1]:first[3]] = 1
    second_canvas[second[0]:second[2], second[1]:second[3]] = 1
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
        # TODO probably no grad is redunant but verify
        for curr_frame in frame_samples:

            rgb_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            t_pil_image = loader(pil_image).float()
            # image = Variable(image, requires_grad=False)
            t_pil_image = t_pil_image.cuda()

            prediction = model([t_pil_image])
            predictions.append(prediction)
            disp_imgs.append(pil_image)  # TODO maybe use it for testing visualization

    w, h = disp_imgs[0].shape
    box_bins = [[prediction[0][0]['boxes'][0].cpu().numpy()]]
    for i in range(1, len(predictions)):
        curr_box = prediction[i][0]['boxes'][0].cpu().numpy()
        found_bin = False
        for j in range(len(box_bins)):
            curr_iou = calc_iou(box_bins[j][0], curr_box, w, h)  # TODO maybe compare all not just 0
            if curr_iou > IOU_THRESHOLD:
                box_bins[j].append(curr_box)
                found_bin = True
                break

        if found_bin is False:
            box_bins.append([curr_box])

        final_box_i = box_bins.index(max(box_bins, key=len))
        final_box = box_bins[final_box_i][0]  # TODO currently taking the first

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


def openpose2motionv2(json_dir, ft_bounding_box, scale=1.0, smooth=True, max_frame=None):
    json_files = sorted(os.listdir(json_dir))
    length = max_frame if max_frame is not None else len(json_files) // 8 * 8
    json_files = json_files[:length]
    json_files = [osp.join(json_dir, x) for x in json_files]

    motion = []
    for path in json_files:
        with open(path) as f:
            jointDict = json.load(f)
            people_poses_arr = jointDict['people']
            for person_pose_info in people_poses_arr:
            # joint = np.array(jointDict['people'][0]['pose_keypoints_2d']).reshape((-1, 3))[:15, :2]
                curr_joint = np.array(person_pose_info['pose_keypoints_2d']).reshape((-1, 3))[:15, :2]
                curr_hips = curr_joint[8]
                if ft_bounding_box[0] < curr_hips[0] < ft_bounding_box[2] and \
                        ft_bounding_box[1] < curr_hips[1] < ft_bounding_box[3]:
                    if len(motion) > 0:
                        curr_joint[np.where(curr_joint == 0)] = motion[-1][np.where(curr_joint == 0)]
                    motion.append(curr_joint)

    for i in range(len(motion) - 1, 0, -1):
        motion[i - 1][np.where(motion[i - 1] == 0)] = motion[i][np.where(motion[i - 1] == 0)]

    motion = np.stack(motion, axis=2)

    if smooth:
        motion = gaussian_filter1d(motion, sigma=2, axis=-1)
    motion = motion * scale
    return motion


def json2npy(data_dir, state_dict, num_samples):
    # preparing model
    model = prepare_model(state_dict)
    clips_dir_fpath = osp.join(data_dir, CLIPS_DIR)
    joints_dir_fpath = osp.join(data_dir, JOINTS_DIR)
    out_dir = osp.join(data_dir, MOTION_DIR)
    vids_kp_dirs = os.listdir(joints_dir_fpath)
    for clip_name in vids_kp_dirs:
        # First we need to find the ft shooter in clip (bounding box)
        curr_clip_fpath = osp.join(clips_dir_fpath, clip_name)
        curr_clip_fpath = f'{curr_clip_fpath}.mp4'
        ft_bounding_box = locate_ft_shooter_in_clip(model, curr_clip_fpath, num_samples=5, num_frames=50)

        # Second we extract all poses into a matrix
        clip_joints_dir_fpath = osp.join(joints_dir_fpath, clip_name)
        joints_json_files = os.listdir(clip_joints_dir_fpath)
        num_frames = len(joints_json_files) # TODO
        motion = openpose2motionv2(clip_joints_dir_fpath, ft_bounding_box, max_frame=42, smooth=False)
        # returned motion shape is (J, 2, max_frame) and belongs to the free throws shooter
        # Here i am saving a matrix representing motion in 42 frames
        save_fpath = osp.join(out_dir, clip_name)
        save_fpath = f'{save_fpath}.npy'
        print(save_fpath)
        np.save(save_fpath, motion)


if __name__ == '__main__':
    args = parse_args()
    json2npy(args.data_dir, args.checkpoint, args.num_samples)