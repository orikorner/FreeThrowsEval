import argparse
import os
import os.path as osp
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default='bbfts_data/train', help='path to data dir of a certain phase')
    parser.add_argument('--motion-dir', type=str, default='motion', help='name of joints dir')
    parser.add_argument('--shot-traj-dir', type=str, default='shot_trajectories', help='name of shot trajectory dir')

    args = parser.parse_args()
    return args


def convert_ball_n_hoop_info_into_np2(objs_info_fpath):
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
    return bball_bb_info, hoop_bb_info


def find_basket_and_save(data_dir, clip_name):
    p_det_dir_fpath = osp.join(data_dir, 'processed_yolo_detections')
    curr_objs_info_fpath = f'{clip_name}.txt'
    curr_objs_info_fpath = osp.join(p_det_dir_fpath, curr_objs_info_fpath)
    _, l_hoop_bb = convert_ball_n_hoop_info_into_np2(curr_objs_info_fpath)
    good_idx = 0
    for i in range(len(l_hoop_bb)):
        if l_hoop_bb[i][0] != 0:
            good_idx = i
            break

    return np.array(l_hoop_bb[good_idx])


def hflip_hoop(in_array, h_dim=1280):
    """[[305, 154], [359, 169]] -> [[975, 154], [921, 169]]"""
    x1 = in_array[0]
    x2 = in_array[2]
    in_array[0] = h_dim - x2
    in_array[2] = h_dim - x1
    return in_array


def hflip_2d(in_array, h_dim=1280):
    for i in range(len(in_array)):
        in_array[i][0] = h_dim - in_array[i][0]

    return in_array


if __name__ == '__main__':
    args = parse_args()

    shot_traj_dir = osp.join(args.data_dir, args.shot_traj_dir)
    motion_dir = osp.join(args.data_dir, args.motion_dir)
    motions = sorted(os.listdir(motion_dir))

    l_hoops_info_str = []
    for i, curr_motion_name in enumerate(motions):

        # if curr_motion_name in ['255.npy', '293.npy', '299.npy', '31.npy', '313.npy', '314.npy', '334.npy']:
        #     l_hoops_info_str.append('0,0,0,0')
        #     continue
        # if curr_motion_name not in ['133.npy']:
        #     continue

        a_hoop_bb = find_basket_and_save(args.data_dir, osp.splitext(curr_motion_name)[0])

        motion_fpath = osp.join(motion_dir, curr_motion_name)
        motion = np.load(motion_fpath)

        shot_traj_fpath = osp.join(shot_traj_dir, curr_motion_name)
        shot_traj = np.load(shot_traj_fpath)

        if shot_traj[0][0] > shot_traj[-1][0]:
            print(f'====== {i}. Flipping: {curr_motion_name} =====')
            shot_traj = hflip_2d(shot_traj)

            for t_i in range(motion.shape[2]):
                motion[:, :, t_i] = hflip_2d(motion[:, :, t_i])

            a_hoop_bb = hflip_hoop(a_hoop_bb.copy())

            np.save(motion_fpath, motion)
            np.save(shot_traj_fpath, shot_traj)

        l_hoops_info_str.append(str(a_hoop_bb.tolist()).strip('[]').replace(' ', ''))

    print(f'====== ALL DONE =====')

    df = pd.DataFrame({'name': motions,
                       'hoop': l_hoops_info_str})
    hoops_info_fpath = osp.join(args.data_dir, 'hoops_info.csv')
    df.to_csv(hoops_info_fpath, index=False)
