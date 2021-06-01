import argparse
import os
import os.path as osp
# from utils.utils import ensure_dir

BBALL_LABEL = 0
HOOP_LABEL = 1


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--trajectories-dir', type=str, help='fpath to dir where all clips trajectories reside')
    parser.add_argument('--out-dir', type=str, help='fpath to dir where we save squeezed trajectories')

    args = parser.parse_args()
    return args


def convert_yolo_pos_fmt_to_act_img_pos(x, y, w, h, img_height=720, img_width=1280):

    l = int((x - w / 2) * img_width)
    r = int((x + w / 2) * img_width)
    t = int((y - h / 2) * img_height)
    b = int((y + h / 2) * img_height)
    if l < 0:
        l = 0
    if r > img_width - 1:
        r = img_width - 1
    if t < 0:
        t = 0
    if b > img_height - 1:
        b = img_height - 1

    return l, t, r, b


def unite_frames_ball_positions_into_file(trajectories_dir, out_dir):
    if not osp.exists(trajectories_dir):
        print('Trajectories dir doesnt exist !')
        return

    # ensure_dir(out_dir)
    l_traj_dirs = os.listdir(trajectories_dir)
    for traj_dir in l_traj_dirs: # Iterating Directories
        vid_out_traj_info_file = osp.join(out_dir, traj_dir)
        vid_out_traj_info_file = f'{vid_out_traj_info_file}.txt'
        traj_dir = osp.join(trajectories_dir, traj_dir)
        l_vid_traj_info_files = os.listdir(traj_dir)
        out_lines = []
        for vid_traj_info in l_vid_traj_info_files:  # Iterating frames of single video
            vid_traj_info = osp.join(traj_dir, vid_traj_info)
            with open(vid_traj_info, 'r') as vid_traj_info_fp:
                max_conf = 0
                curr_out_line = '0 0 0 0 0 0\n'
                for i, line in enumerate(vid_traj_info_fp):
                    if line.strip():
                        # Check that ball was found (not an empty line)
                        lbl_id, x, y, w, h, conf = [float(x) for x in line.split(' ')]
                        x1, y1, x2, y2 = convert_yolo_pos_fmt_to_act_img_pos(x, y, w, h)
                        if conf > 0.75 and conf > max_conf:
                            curr_out_line = f'{int(lbl_id)} {x1} {y1} {x2} {y2}\n'

                out_lines.append(curr_out_line)

        with open(vid_out_traj_info_file, 'w+') as vid_out_traj_info_fp:
            vid_out_traj_info_fp.writelines(out_lines)


if __name__ == '__main__':
    args = parse_args()

    unite_frames_ball_positions_into_file(args.trajectories_dir, args.out_dir)
