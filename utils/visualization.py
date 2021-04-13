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
import os.path as osp
# from utils.operators import normalize_motion_inv, trans_motion_inv


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--joints', type=str, help='full path to joints file as .npy')
    parser.add_argument('--in-vid', type=str, help='full path to clip as .mp4')
    parser.add_argument('--out-vid', type=str, help='full path to output clip dir')

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


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, vid_names, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    dummy_mat = [(np.random.rand(28, 28) - 0.5) / 0.5 for ii in range(len(labels))]
    # dummy_mat = [np.random.randn(28, 28) for ii in range(len(labels))]
    # dummy_mat = np.stack(dummy_mat, axis=0)
    # dummy_mat = torch.from_numpy(dummy_mat)
    classes = ('Miss', 'Score')
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(len(labels)):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        # matplotlib_imshow(dummy_mat[idx], one_channel=True)
        plt.imshow(dummy_mat[idx], cmap="Greys")
        ax.set_title("{0}, {1:.1f}%\n(label: {2})\n{3}".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]],
            vid_names[idx]),
                    color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


# def write_pose_to_video(in_video_fpath, out_video_fpath, frames):
#     """
#     Writes frames to an mp4 video file
#     :param file_path: Path to output video, must end with .mp4
#     :param frames: List of PIL.Image objects
#     :param fps: Desired frame rate
#     """
#
#     capture = cv2.VideoCapture(in_video_fpath)
#     width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = int(capture.get(cv2.CAP_PROP_FPS))
#
#     data = torch.FloatTensor([frames]).cuda()
#     video_out = cv2.VideoWriter(out_video_fpath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
#
#     for i in tqdm(range(length)):
#         ok, frame = capture.read()
#         frame = torch.FloatTensor([frame]) / 127.5 - 1.0  # (L, H, W, 3)
#         frame = frame.permute(3, 0, 1, 2).unsqueeze(0).cuda()  # (1, 3, L, H, W)
#         wm_frame = self.encoder(frame, data)  # (1, 3, L, H, W)
#         wm_frame = torch.clamp(wm_frame, min=-1.0, max=1.0)
#         wm_frame = (
#                 (wm_frame[0, :, 0, :, :].permute(1, 2, 0) + 1.0) * 127.5
#         ).detach().cpu().numpy().astype("uint8")
#         video_out.write(wm_frame)
#
#     video_out.release()
#     capture.release()
#     cv2.destroyAllWindows()


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


def joints2image(joints_position, colors, transparency=False, H=512, W=512, nr_joints=49, imtype=np.uint8):
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
            cv2.circle(canvas, (int(all_peaks[i][0]), int(all_peaks[i][1])), 2, joint_colors[i], thickness=2)

    if _limb:
        stickwidth = 2

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


def motion2video(motion, h, w, save_path, colors, transparency=False, motion_tgt=None, fps=25, save_frame=False):
    nr_joints = motion.shape[0]
    videowriter = imageio.get_writer(save_path, fps=fps)
    vlen = motion.shape[-1]

    if save_frame:
        frames_dir = save_path[:-4] + '-frames'
        utils.ensure_dir(frames_dir)
    for i in tqdm(range(vlen)):
        [img, img_cropped] = joints2image(motion[:, :, i], colors, transparency, H=h, W=w, nr_joints=nr_joints)
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
#
#
# def visulize_motion_in_training(output, mean_pose, std_pose, nr_visual=4, H=512, W=512):
#     ret = {}
#
#     motion = output.detach().cpu().numpy()
#     inds = np.linspace(0, motion.shape[1] - 1, nr_visual, dtype=int)
#     motion = motion[:, inds]
#     motion = motion.reshape(-1, 2, motion.shape[-1])
#     motion = normalize_motion_inv(motion, mean_pose, std_pose)
#     peaks = trans_motion_inv(motion)
#
#     heatmaps = []
#     for i in range(peaks.shape[2]):
#         skeleton = pose2im_all(peaks[:, :, i], H, W)
#         heatmaps.append(skeleton)
#     heatmaps = np.stack(heatmaps).transpose((0, 3, 1, 2)) / 255.0
#     # ret[k] = heatmaps
#
#     return ret


if __name__ == '__main__':
    args = parse_args()
    motion = np.load(args.joints)
    # for i in range(42):
    #     print(motion[8, :, i])
    capture = cv2.VideoCapture(args.in_vid)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    color1 = hex2rgb('#a50b69#b73b87#db9dc3')
    color2 = hex2rgb('#4076e0#40a7e0#40d7e0')
    color3 = hex2rgb('#ff8b06#ffb431#ffcd9d')
    colors = [color1, color2, color3]
    motion2video(motion, height, width, args.out_vid, color1, transparency=False, motion_tgt=None, fps=fps, save_frame=False)
