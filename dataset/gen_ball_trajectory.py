import argparse
import os
import os.path as osp
# from utils.utils import ensure_dir

BBALL_LABEL = 0
HOOP_LABEL = 1
OBJ_NOT_FOUND = '0 0 0 0 0'
OBJS_NOT_FOUND = '0 0 0 0 0,0 0 0 0 0\n'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default='bbfts_data/train', help='path to data dir of a certain phase')
    parser.add_argument('--detections-dir', type=str, default='yolo_detections', help='dir of yolo detections')
    parser.add_argument('--out-dir', type=str, default='processed_yolo_detections', help='dir of processed detections')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Object detection confidence threshold')

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


def get_frame_index_by_file(fpath):
    return int(osp.splitext(osp.basename(fpath))[0].split('_')[1])


def unite_frames_ball_positions_into_file(detections_dir, out_dir, conf_threshold):
    if not osp.exists(detections_dir):
        print('Detections dir doesnt exist !')
        return

    # ensure_dir(out_dir)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    l_det_dirs = sorted(os.listdir(detections_dir))
    for det_dir in l_det_dirs:  # Iterating Directories
        # if int(det_dir) <= 862:
        #     continue
        vid_out_det_info_file = osp.join(out_dir, det_dir)
        vid_out_det_info_file = f'{vid_out_det_info_file}.txt'
        det_dir = osp.join(detections_dir, det_dir)
        l_vid_det_info_files = sorted(os.listdir(det_dir))
        out_lines = []
        prev_ball_found_frame_idx = 0
        for vid_det_info in l_vid_det_info_files:  # Iterating frames of single video
            vid_det_info = osp.join(det_dir, vid_det_info)

            # First we need to fill lines from frames where ball and hoop were not located
            # TODO the end is not filled, but it should be fine since we assume shot release will be found much sooner
            curr_ball_found_frame_index = get_frame_index_by_file(vid_det_info)
            if curr_ball_found_frame_index > (prev_ball_found_frame_idx + 1):
                for j in range(1, curr_ball_found_frame_index - prev_ball_found_frame_idx):
                    out_lines.append(OBJS_NOT_FOUND)
            prev_ball_found_frame_idx = curr_ball_found_frame_index

            with open(vid_det_info, 'r') as vid_det_info_fp:
                max_conf_ball = 0
                max_conf_hoop = 0
                curr_ball_info = OBJ_NOT_FOUND
                curr_hoop_info = OBJ_NOT_FOUND
                for i, line in enumerate(vid_det_info_fp):
                    if line.strip():
                        # Not an empty line
                        obj_id, x, y, w, h, conf = [float(x) for x in line.split(' ')]
                        x1, y1, x2, y2 = convert_yolo_pos_fmt_to_act_img_pos(x, y, w, h)
                        if obj_id == 0:
                            # Ball object
                            if conf > conf_threshold and conf > max_conf_ball:
                                max_conf_ball = conf
                                curr_ball_info = f'{int(obj_id)} {x1} {y1} {x2} {y2}'
                        elif obj_id == 1:
                            # Hoop object
                            if conf > max_conf_hoop:
                                max_conf_hoop = conf
                                curr_hoop_info = f'{int(obj_id)} {x1} {y1} {x2} {y2}'
                        else:
                            raise ValueError(f'!!! Invalid obj id found: {obj_id}')
                curr_out_line = f'{curr_ball_info},{curr_hoop_info}\n'
                out_lines.append(curr_out_line)

        with open(vid_out_det_info_file, 'w+') as vid_out_det_info_fp:
            vid_out_det_info_fp.writelines(out_lines)


if __name__ == '__main__':
    args = parse_args()

    detections_dir = osp.join(args.data_dir, args.detections_dir)
    out_dir = osp.join(args.data_dir, args.out_dir)
    unite_frames_ball_positions_into_file(detections_dir, out_dir, args.conf_threshold)
