import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


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

# from utils.operators import normalize_motion_inv, trans_motion_inv
# import math
# import cv2


# def pose2im_all(all_peaks, H=512, W=512):
#     limbSeq = [[1, 2], [2, 3], [3, 4],                       # right arm
#                [1, 5], [5, 6], [6, 7],                       # left arm
#                [8, 9], [9, 10], [10, 11],                    # right leg
#                [8, 12], [12, 13], [13, 14],                  # left leg
#                [1, 0],                                       # head/neck
#                [1, 8],                                       # body,
#                ]
#
#     limb_colors = [[0, 60, 255], [0, 120, 255], [0, 180, 255],
#                     [180, 255, 0], [120, 255, 0], [60, 255, 0],
#                     [170, 255, 0], [85, 255, 0], [0, 255, 0],
#                     [255, 170, 0], [255, 85, 0], [255, 0, 0],
#                     [0, 85, 255],
#                     [0, 0, 255],
#                    ]
#
#     joint_colors = [[85, 0, 255], [0, 0, 255], [0, 60, 255], [0, 120, 255], [0, 180, 255],
#                     [180, 255, 0], [120, 255, 0], [60, 255, 0], [0, 0, 255],
#                     [170, 255, 0], [85, 255, 0], [0, 255, 0],
#                     [255, 170, 0], [255, 85, 0], [255, 0, 0],
#                     ]
#
#     image = pose2im(all_peaks, limbSeq, limb_colors, joint_colors, H, W)
#     return image
#
#
# def pose2im(all_peaks, limbSeq, limb_colors, joint_colors, H, W, _circle=True, _limb=True, imtype=np.uint8):
#     canvas = np.zeros(shape=(H, W, 3))
#     canvas.fill(255)
#
#     if _circle:
#         for i in range(len(joint_colors)):
#             cv2.circle(canvas, (int(all_peaks[i][0]), int(all_peaks[i][1])), 2, joint_colors[i], thickness=2)
#
#     if _limb:
#         stickwidth = 2
#
#         for i in range(len(limbSeq)):
#             limb = limbSeq[i]
#             cur_canvas = canvas.copy()
#             point1_index = limb[0]
#             point2_index = limb[1]
#
#             if len(all_peaks[point1_index]) > 0 and len(all_peaks[point2_index]) > 0:
#                 point1 = all_peaks[point1_index][0:2]
#                 point2 = all_peaks[point2_index][0:2]
#                 X = [point1[1], point2[1]]
#                 Y = [point1[0], point2[0]]
#                 mX = np.mean(X)
#                 mY = np.mean(Y)
#                 # cv2.line()
#                 length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
#                 angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
#                 polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
#                 cv2.fillConvexPoly(cur_canvas, polygon, limb_colors[i])
#                 canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
#
#     return canvas.astype(imtype)
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