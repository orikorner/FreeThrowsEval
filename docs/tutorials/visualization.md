# Tutorial: Visualization


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


2.  Create Shot info clip based on our findings (Shooter Motion + Shot Trajectory + Hoop), Run:
python utils/visualization.py --data-dir bbfts_data/<'phase'> --out-dir visualizations/<'phase'>