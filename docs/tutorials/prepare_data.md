# Tutorial: Preparing bbfts dataset

**** First of all ****
We assume we have the mp4 clips and OpenPose was used to extract joints.
This should be the folder structure:

FreeThrowsEval
├── dataset
│   ├── json2npy.py
├── bbfts_data
│   ├── bbfts_labels.csv
│   ├── train
│   │   ├── clips # Containing videos (.mp4)
│   │   ├── joints # Containing all joints of all people (.json)
│   ├── test
│   │   ├── clips # Containing videos (.mp4)
│   │   ├── joints # Containing all joints of all people (.json)


1.  For every clip we will find the joints belonging to the shooter,
    and save it as a motion matrix (np.array) of shape (J, 2, num_frames):
Verify that mask_rcnn_latest_e9.pth is in the checkpoints directory, then run:
python dataset/json2npy.py --data-dir bbfts_data/train --checkpoint checkpoints/mask_rcnn_latest_e9.pth


2.  For every clip we will find the frame index where the ball was released, and add it to labels file.
    Also, the shot trajectory is produced and saved as numpy matrix.
    For this we need the yolov5 detections. meaning folder structure:

FreeThrowsEval
├── bbfts_data
│   ├── bbfts_labels.csv
│   ├── train
│   │   ├── clips # Containing videos (.mp4)
│   │   ├── joints # Containing all joints of all people (.json)
│   │   ├── motion # Containing the motion of the ft shooter (obtained in last section)
│   │   ├── yolo_detections # Containing yolov5 inference on all the clips (frame by frame)
│   ├── test
│   │   ├── clips # Containing videos (.mp4)
│   │   ├── joints # Containing all joints of all people (.json)
│   │   ├── motion # Containing the motion of the ft shooter (obtained in last section)
│   │   ├── yolo_detections # Containing yolov5 inference on all the clips (frame by frame)

    2.1. Preprocess the detections and save in shot_trajectories folder, run:
    python dataset/gen_ball_trajectory.py --data-dir bbfts_data/<phase>

    2.2. Find the shot index and produce shot trajectory, run:
    python dataset/gen_shot_trajectory.py --data-dir bbfts_data --phase <phase>

At this point the folder structure is:
FreeThrowsEval
├── bbfts_data
│   ├── bbfts_labels.csv <- Updated
│   ├── train
│   │   ├── clips # Containing videos (.mp4)
│   │   ├── joints # Containing all joints of all people (.json)
│   │   ├── motion # Containing the motion of the ft shooter (obtained in last section)
│   │   ├── yolo_detections # Containing yolov5 inference on all the clips (frame by frame)
│   │   ├── bball_trajectories # Containing the pre processing done over yolo_detections
│   │   ├── shot_trajectories # Containing ball trajectory from shot frame to hoop impact frame
│   ├── test
│   │   ├── clips # Containing videos (.mp4)
│   │   ├── joints # Containing all joints of all people (.json)
│   │   ├── motion # Containing the motion of the ft shooter (obtained in last section)
│   │   ├── yolo_detections # Containing yolov5 inference on all the clips (frame by frame)
│   │   ├── bball_trajectories # Containing the pre processing done over yolo_detections
│   │   ├── shot_trajectories # Containing ball trajectory from shot frame to hoop impact frame

