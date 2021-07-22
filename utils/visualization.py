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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default='bbfts_data/train', help='path to data dir of a certain phase')
    parser.add_argument('--motion-dir', type=str, default='motion', help='name of joints dir')
    parser.add_argument('--clips-dir', type=str, default='clips', help='name of clips dir')
    parser.add_argument('--out-dir', type=str, default='visualizations/train', help='full path to output dir')
    parser.add_argument('--detections-dir', type=str, default='processed_yolo_detections', help='name of ball and hoop locations processed dir')
    parser.add_argument('--shot-traj-dir', type=str, default='shot_trajectories', help='name of shot trajectory dir')
    parser.add_argument('--labels-info', type=str, default='bbfts_data/bbfts_labels.csv', help='full path to labels file')


    args = parser.parse_args()
    return args


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(outputs, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(outputs, 1)
    preds = np.squeeze(preds_tensor.cpu().clone().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, outputs)]


def get_predictions(outputs, vid_names, images, labels, w_print=True):
    preds, probs = images_to_probs(outputs, images)
    labels = labels.reshape(-1).numpy()
    correct = np.logical_not(np.logical_xor(np.array(preds), labels)).sum()
    num_ones = 0
    for i in range(len(labels)):
        if preds[i] == 1 and labels[i] == 1:
            num_ones += 1

    if w_print:
        print()
        # print(f'Confidences: {["{0:0.2f}".format(x) for x in probs]}')
        print(f'Predcts:     {preds}')
        print(f'Lables:      {labels}')

        print(f'Accuracy: {correct}/{len(labels)} ({correct / len(labels):.3}) Correct Made Shots: {num_ones}/{correct}')
    return preds, probs, correct


def plot_classes_preds(outputs, vid_names, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs, correct = get_predictions(outputs, vid_names, images, labels)

    dummy_mat = [(np.random.rand(28, 28) - 0.5) / 0.5 for ii in range(len(labels))]
    # dummy_mat = [np.random.randn(28, 28) for ii in range(len(labels))]
    # dummy_mat = np.stack(dummy_mat, axis=0)
    # dummy_mat = torch.from_numpy(dummy_mat)
    classes = ('Miss', 'Score')
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(len(labels)):
        ax = fig.add_subplot(1, len(images), idx+1, xticks=[], yticks=[])
        # matplotlib_imshow(dummy_mat[idx], one_channel=True)
        plt.imshow(dummy_mat[idx], cmap="Greys")
        ax.set_title("{0}, {1:.1f}%\n(label: {2})\n{3}".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]],
            vid_names[idx]),
                    color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def bounding_box(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox


def hex2rgb(hex, number_of_colors=3):
    h = hex
    rgb = []
    for i in range(number_of_colors):
        h = h.lstrip('#')
        hex_color = h[0:6]
        rgb_color = [int(hex_color[i:i+2], 16) for i in (0, 2 ,4)]
        rgb.append(rgb_color)
        h = h[6:]

    return rgb


def rgb2rgba(color):
    return (color[0], color[1], color[2], 255)


def joints2image(joints_position, colors, transparency=False, H=512, W=512, nr_joints=49, shot_released=False, imtype=np.uint8):
    nr_joints = joints_position.shape[0]

    if nr_joints == 49: # full joints(49): basic(15) + eyes(2) + toes(2) + hands(30)
        limbSeq = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], \
                   [8, 9], [8, 13], [9, 10], [10, 11], [11, 12], [13, 14], [14, 15], [15, 16],
                   ]#[0, 17], [0, 18]] #ignore eyes

        L = rgb2rgba(colors[0]) if transparency else colors[0]
        M = rgb2rgba(colors[1]) if transparency else colors[1]
        R = rgb2rgba(colors[2]) if transparency else colors[2]

        colors_joints = [M, M, L, L, L, R, R,
                  R, M, L, L, L, L, R, R, R,
                  R, R, L] + [L] * 15 + [R] * 15

        colors_limbs = [M, L, R, M, L, L, R,
                  R, L, R, L, L, L, R, R, R,
                  R, R]
    elif nr_joints == 15 or nr_joints == 17: # basic joints(15) + (eyes(2))
        limbSeq = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7],
                   [8, 9], [8, 12], [9, 10], [10, 11], [12, 13], [13, 14]]
                    # [0, 15], [0, 16] two eyes are not drawn

        L = rgb2rgba(colors[0]) if transparency else colors[0]
        M = rgb2rgba(colors[1]) if transparency else colors[1]
        R = rgb2rgba(colors[2]) if transparency else colors[2]

        colors_joints = [M, M, L, L, L, R, R,
                         R, M, L, L, L, R, R, R]

        colors_limbs = [M, L, R, M, L, L, R,
                        R, L, R, L, L, R, R]
    else:
        raise ValueError("Only support number of joints be 49 or 17 or 15")

    if transparency:
        canvas = np.zeros(shape=(H, W, 4))
    else:
        canvas = np.ones(shape=(H, W, 3)) * 255
        if shot_released:
            canvas[-10:, :, 1:-1] = 0  # Set Bottom frame line in Red
            canvas[:10, :, 1:-1] = 0  # Set Bottom frame line in Red
            canvas[:, -10:, 1:-1] = 0  # Set Bottom frame line in Red
            canvas[:, :10, 1:-1] = 0  # Set Bottom frame line in Red

    hips = joints_position[8]
    neck = joints_position[1]
    torso_length = ((hips[1] - neck[1]) ** 2 + (hips[0] - neck[0]) ** 2) ** 0.5

    head_radius = int(torso_length/4.5)
    end_effectors_radius = int(torso_length/15)
    end_effectors_radius = 7
    joints_radius = 7

    cv2.circle(canvas, (int(joints_position[0][0]), int(joints_position[0][1])), head_radius, colors_joints[0], thickness=-1)

    for i in range(1, len(colors_joints)):
        if i in (17, 18):
            continue
        elif i > 18:
            radius = 2
        else:
            radius = joints_radius
        cv2.circle(canvas, (int(joints_position[i][0]),int(joints_position[i][1])), radius, colors_joints[i], thickness=-1)

    stickwidth = 2

    for i in range(len(limbSeq)):
        limb = limbSeq[i]
        cur_canvas = canvas.copy()
        point1_index = limb[0]
        point2_index = limb[1]

        #if len(all_peaks[point1_index]) > 0 and len(all_peaks[point2_index]) > 0:
        point1 = joints_position[point1_index]
        point2 = joints_position[point2_index]
        X = [point1[1], point2[1]]
        Y = [point1[0], point2[0]]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        alpha = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(alpha), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors_limbs[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        bb = bounding_box(canvas)
        canvas_cropped = canvas[:,bb[2]:bb[3], :]

    return [canvas.astype(imtype), canvas_cropped.astype(imtype)]


def pose2im_all(all_peaks, H=512, W=512):
    limbSeq = [[1, 2], [2, 3], [3, 4],                       # right arm
               [1, 5], [5, 6], [6, 7],                       # left arm
               [8, 9], [9, 10], [10, 11],                    # right leg
               [8, 12], [12, 13], [13, 14],                  # left leg
               [1, 0],                                       # head/neck
               [1, 8],                                       # body,
               ]

    limb_colors = [[0, 60, 255], [0, 120, 255], [0, 180, 255],
                    [180, 255, 0], [120, 255, 0], [60, 255, 0],
                    [170, 255, 0], [85, 255, 0], [0, 255, 0],
                    [255, 170, 0], [255, 85, 0], [255, 0, 0],
                    [0, 85, 255],
                    [0, 0, 255],
                   ]

    joint_colors = [[85, 0, 255], [0, 0, 255], [0, 60, 255], [0, 120, 255], [0, 180, 255],
                    [180, 255, 0], [120, 255, 0], [60, 255, 0], [0, 0, 255],
                    [170, 255, 0], [85, 255, 0], [0, 255, 0],
                    [255, 170, 0], [255, 85, 0], [255, 0, 0],
                    ]

    image = pose2im(all_peaks, limbSeq, limb_colors, joint_colors, H, W)
    return image


def pose2im(all_peaks, limbSeq, limb_colors, joint_colors, H, W, _circle=True, _limb=True, imtype=np.uint8):
    canvas = np.zeros(shape=(H, W, 3))
    canvas.fill(255)

    if _circle:
        for i in range(len(joint_colors)):
            cv2.circle(canvas, (int(all_peaks[i][0]), int(all_peaks[i][1])), 2, joint_colors[i], thickness=1)

    if _limb:
        stickwidth = 1

        for i in range(len(limbSeq)):
            limb = limbSeq[i]
            cur_canvas = canvas.copy()
            point1_index = limb[0]
            point2_index = limb[1]

            if len(all_peaks[point1_index]) > 0 and len(all_peaks[point2_index]) > 0:
                point1 = all_peaks[point1_index][0:2]
                point2 = all_peaks[point2_index][0:2]
                X = [point1[1], point2[1]]
                Y = [point1[0], point2[0]]
                mX = np.mean(X)
                mY = np.mean(Y)
                # cv2.line()
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, limb_colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas.astype(imtype)


def get_hoop_plot_info(hoop_bb):
    bball_center = np.array([0.5 * (hoop_bb[0] + hoop_bb[2]), 0.5 * (hoop_bb[1] + hoop_bb[3])])
    return int(hoop_bb[0]), int(bball_center[1]), int(hoop_bb[2]), int(bball_center[1])


def motion2video(motion, h, w, save_path, colors, shot_rel_frame=-1, shot_traj=None, hoop_bbs=None, transparency=False, motion_tgt=None, fps=25, save_frame=False):
    shot_released = False
    nr_joints = motion.shape[0]
    videowriter = imageio.get_writer(save_path, fps=fps)
    vlen = motion.shape[-1]
    k = 0
    if save_frame:
        frames_dir = save_path[:-4] + '-frames'
        utils.ensure_dir(frames_dir)
    for i in tqdm(range(vlen)):
        if i >= shot_rel_frame:
            shot_released = True
        [img, img_cropped] = joints2image(motion[:, :, i], colors, transparency, H=h, W=w, nr_joints=nr_joints, shot_released=shot_released)
        if shot_released and k < len(shot_traj):
            cv2.circle(img, (int(shot_traj[k][0]), int(shot_traj[k][1])), 5, [255, 0, 0], thickness=4)
            k += 1
        # Draw Hoop
        if len(hoop_bbs) > i:
            x1, y1, x2, y2 = get_hoop_plot_info(hoop_bbs[i])
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)
        if motion_tgt is not None:
            [img_tgt, img_tgt_cropped] = joints2image(motion_tgt[:, :, i], colors, transparency, H=h, W=w, nr_joints=nr_joints)
            img_ori = img.copy()
            img = cv2.addWeighted(img_tgt, 0.3, img_ori, 0.7, 0)
            img_cropped = cv2.addWeighted(img_tgt, 0.3, img_ori, 0.7, 0)
            bb = bounding_box(img_cropped)
            img_cropped = img_cropped[:, bb[2]:bb[3], :]
        if save_frame:
            save_image(img_cropped, osp.join(frames_dir, "%04d.png" % i))
        videowriter.append_data(img)
    videowriter.close()


def extract_hoop_info_into_np(objs_info_fpath):
    """ Takes a single hoop and ball locations info file and parses is into a numpy array.
    every line in the file represents a frame, and every line is of format:
    obj_id x1 y1 x2 y2,obj_id x1 y1 x2 y2,
    we return 2 numpy arrays (ball and hoop) with the coordinates separated by a comma
    """
    hoop_bb_info = []
    with open(objs_info_fpath, 'r') as objs_info_fp:
        for i, line in enumerate(objs_info_fp):
            if not line.strip():
                # Check if ball not found (empty line)
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
                    hoop_bb_info.append(obj1)
                else:
                    hoop_bb_info.append(obj2)

    return np.array(hoop_bb_info)


def flip_horizontaly(in_array):
    for i in range(len(in_array)):
        in_array[i][0] = 1280 - in_array[i][0]
    return in_array


if __name__ == '__main__':
    args = parse_args()

    shot_rel_df = pd.read_csv(args.labels_info, header=0)

    utils.ensure_dir(args.out_dir)
    shot_traj_dir = osp.join(args.data_dir, args.shot_traj_dir)
    detections_dir = osp.join(args.data_dir, args.detections_dir)
    motion_dir = osp.join(args.data_dir, args.motion_dir)
    motions = os.listdir(motion_dir)
    clips_dir = osp.join(args.data_dir, args.clips_dir)
    clips = [x for x in os.listdir(clips_dir) if x.endswith('.mp4')]

    assert len(motions) == len(clips)

    color1 = hex2rgb('#a50b69#b73b87#db9dc3')
    color2 = hex2rgb('#4076e0#40a7e0#40d7e0')
    color3 = hex2rgb('#ff8b06#ffb431#ffcd9d')
    colors = [color1, color2, color3]

    for i, curr_clip_name in enumerate(clips):
        curr_motion_name = curr_clip_name.split(".")[0]
        # if int(curr_motion_name) != 109:
        #     continue
        curr_shot_rl_frame = shot_rel_df.loc[shot_rel_df['video_name'] == int(curr_motion_name)]['shot_frame'].item()
        a_hoop_bb = extract_hoop_info_into_np(osp.join(detections_dir, f'{curr_motion_name}.txt'))
        curr_motion_name = f'{curr_motion_name}.npy'

        curr_out_name = osp.join(args.out_dir, curr_clip_name)

        shot_traj = np.load(osp.join(shot_traj_dir, curr_motion_name))

        motion = np.load(osp.join(motion_dir, curr_motion_name))

        # if shot_traj[0][0] > shot_traj[-1][0]:
        #     shot_traj = flip_horizontaly(shot_traj)
        #
        #     for t_i in range(motion.shape[2]):
        #         motion[:, :, t_i] = flip_horizontaly(motion[:, :, t_i])
        #
        #     a_hoop_bb = a_hoop_bb.reshape((-1, 2, 2))
        #     for t_j in range(a_hoop_bb.shape[0]):
        #         a_hoop_bb[t_j, :, :] = flip_horizontaly(a_hoop_bb[t_j, :, :])
        #     a_hoop_bb = a_hoop_bb.reshape((-1, 4))

        capture = cv2.VideoCapture(osp.join(clips_dir, curr_clip_name))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))

        if width > 1000 and height > 700:
            if fps < 24 or fps > 25:
                print(f'{curr_clip_name} - FPS: {fps}')

            motion2video(motion, height, width, curr_out_name, color1,
                         shot_rel_frame=curr_shot_rl_frame, shot_traj=shot_traj, hoop_bbs=a_hoop_bb,
                         transparency=False, motion_tgt=None, fps=fps, save_frame=False)
        else:
            print(f'{curr_clip_name} - W: {width} , H: {height}')

        capture.release()
        print(f'====== Finished {i} ======')
