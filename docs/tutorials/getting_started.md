# Tutorial: Getting Started

1.  Follow the steps described in prepare_data.md. At this point the folder structure should be:

FreeThrowsEval
├── bbfts_data
│   ├── bbfts_labels.csv
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


2.  Train model, Run:
python train.py --name <name>
