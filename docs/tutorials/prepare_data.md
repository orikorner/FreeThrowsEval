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


2. For every clip we will find the frame index where the ball was released, and save it as csv file
    % mkdir data
    % mkdir checkpoints
    % cd data
    % mkdir basketball3
    % cd basketball3
    % mkdir extended_clips
    % mkdir rawframes
    % cd rawframes
    % mkdir flows
    % cd flows
    % mkdir train
    % mkdir test
    % mkdir validation
    % cd ../../
    % mkdir rgb_feat
    % mkdir flow_feat
    % cd flow_feat
    % mkdir train
    % mkdir test
    % mkdir validation
    % cd ..
    % mkdir processed_rgb_n_flow_feat
    % cd processed_rgb_n_flow_feat
    % mkdir train
    % mkdir test
    % mkdir validation
    % cd ..
    % cd extended_clips
    % mkdir extended_clips_by_subset
    % cd extended_clips_by_subset
    % mkdir test
    % mkdir train
    % mkdir validation
    % cd ../../../../tools/data
    % mkdir basketball3

This would result in remaining in /mmaction directory and the folder structure is:

mmaction2
├── mmaction
├── tools
│   ├── data
│       ├── basketball3
├── configs
├── data
│   ├── basketball3
│   │   ├── extended_clips
│   │   │   ├── extended_clips_by_subset
│   │   │   │   ├── train
│   │   │   │   ├── test
│   │   │   │   ├── validation
│   │   ├── rawframes
│   │   │   ├── flows
│   │   │   │   ├── train
│   │   │   │   ├── test
│   │   │   │   ├── validation
│   │   ├── rgb_feat
│   │   ├── flow_feat
│   │   │   ├── train
│   │   │   ├── test
│   │   │   ├── validation
│   │   ├── processed_rgb_n_flow_feat
│   │   │   ├── train
│   │   │   ├── test
│   │   │   ├── validation

2. Now we place .mp4 clips in extended_clips_by_subset in train/test/validation directories

3. Now we need to extract frames from extended_clips. From tools/data directory run:
python build_rawframes.py ../../data/basketball3/extended_clips/extended_clips_by_subset/test ../../data/basketball3/rawframes/test --level 1 --flow-type tvl1 --ext mp4 --task rgb --use-opencv --new-width 340 --new-height 256
python build_rawframes.py ../../data/basketball3/extended_clips/extended_clips_by_subset/train ../../data/basketball3/rawframes/train --level 1 --flow-type tvl1 --ext mp4 --task rgb --use-opencv --new-width 340 --new-height 256
python build_rawframes.py ../../data/basketball3/extended_clips/extended_clips_by_subset/validation ../../data/basketball3/rawframes/validation --level 1 --flow-type tvl1 --ext mp4 --task rgb --use-opencv --new-width 340 --new-height 256

At this point the rawframes directory changes such that:
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── basketball3
│   │   ├── rgb_feat
│   │   ├── flow_feat
│   │   ├── processed_rgb_n_flow_feat
│   │   ├── extended_clips
│   │   │   ├── extended_clips_by_subset
│   │   │   │   ├── train
│   │   │   │   |   ├── vid_name1.mp4
│   │   │   │   |   ├── ..
│   │   │   │   ├── test
│   │   │   │   |   ├── vid_name2.mp4
│   │   │   │   |   ├── ..
│   │   │   │   ├── validation
│   │   │   │   |   ├── vid_name3.mp4
│   │   │   │   |   ├── ..
│   │   ├── rawframes
│   │   │   ├── flows
│   │   │   ├── train
│   │   │   |   ├── vid_name1
│   │   │   |   |   ├── img_00001.jpg
│   │   │   |   |   ├── ..
│   │   │   ├── test
│   │   │   │   ├── vid_name2
│   │   │   │   |   ├── img_00001.jpg
│   │   │   │   |   ├── ..
│   │   │   ├── validation
│   │   │   |   ├── vid_name3
│   │   │   |   |   ├── img_00001.jpg
│   │   │   |   |   ├── ..

4. The following files should be in tools/data/basketball3:
    - bb4_create_video_rawframes_annot_txt_file.py
Run:
python bb4_create_video_rawframes_annot_txt_file.py

Which will produce the annotation txt files for train, test and validation data by walking through their dirs and counting frames.
It is needed for the TSN model for feature extraction (RGB Stream only since).
At this point the folder structure contains them:

├── data
│   ├── basketball3
│   │   ├── bb4_videos_rgb_rf_train.txt
│   │   ├── bb4_videos_rgb_rf_test.txt
│   │   ├── bb4_videos_rgb_rf_validation.txt

5. Extract features using TSN (RGB Stream only) - go to tools/data/basketball3 directory, then:
    5.1. bb4_tsn_rgb_config.py config should be in the config/recognition/tsn directory.
    5.2. RGB tsn checkpoint should be in the checkpoints directory (e.g checkpoints/epoch_12.pth) TODO
    5.3. Extract training/test/validation data features: TODO ckpt name
        python bb4_feature_extractor.py --data-prefix ../../../data/basketball3 --data-list ../../../data/basketball3/bb4_videos_rgb_rf_train.txt --output-prefix ../../../data/basketball3/rgb_feat/train --modality RGB --ckpt ../../../checkpoints/epoch_12.pth --frame-interval 6
        python bb4_feature_extractor.py --data-prefix ../../../data/basketball3 --data-list ../../../data/basketball3/bb4_videos_rgb_rf_validation.txt --output-prefix ../../../data/basketball3/rgb_feat/validation --modality RGB --ckpt ../../../checkpoints/epoch_12.pth --frame-interval 6
        python bb4_feature_extractor.py --data-prefix ../../../data/basketball3 --data-list ../../../data/basketball3/bb4_videos_rgb_rf_test.txt --output-prefix ../../../data/basketball3/rgb_feat/test --modality RGB --ckpt ../../../checkpoints/epoch_12.pth --frame-interval 6

At this point we have extract features from RGB frames only, where each feature is of dimension Nx200 where N=vid_num_frames/6.
And the folder structure is:
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── basketball3
│   │   ├── processed_rgb_n_flow_feat
│   │   ├── extended_clips
│   │   ├── rawframes
│   │   ├── flow_feat
│   │   ├── rgb_feat
│   │   |   ├── train
│   │   │   │   ├── fb1_u2qAx-8ACOA.pkl
│   │   │   │   :
│   │   |   ├── test
│   │   │   │   ├── fb1_6WPSYIdbhF8.pkl
│   │   │   │   :
│   │   |   ├── validation
│   │   │   │   ├── fb1_vci_9DF3VDU.pkl
│   │   │   │   :

6. Extract Optical Flow features (see bb4_tsn_train.md) # TODO

7. Post Process RGB and Flow features into a single Fused feature in dimension 100x400 using linear interpolation and
   concatenation in temporal dimension (see bb3_train_n_test_resnet_tsn.md)

At this point the dataset for BMN is ready and the folder structure is:
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── basketball3
│   │   ├── extended_clips
│   │   ├── rawframes
│   │   ├── rgb_feat
│   │   ├── flow_feat
│   │   ├── processed_rgb_n_flow_feat
│   │   |   ├── train
│   │   │   │   ├── fb1_u2qAx-8ACOA.csv
│   │   │   │   :
│   │   |   ├── test
│   │   │   │   ├── fb1_6WPSYIdbhF8.csv
│   │   │   │   :
│   │   |   ├── validation
│   │   │   │   ├── fb1_vci_9DF3VDU.csv
│   │   │   │   :

8. Now we need to prepare a labels file for training BMN:
    8.1. Create videos info file (bb4_video_info.csv):
        - Go to tools/data/basketball3 directory.
        - Verify videos in extended_clips_by_subset/train (also test, validation) directory.
        - Run: python bb4_bmn_create_rawframes_info_file.py
    8.2. Create a json file for each subset (train/test/valid).
        - bb4_proposals_gt.json in dataset main dir. # TODO name (proposals_info)
        - bb4_video_info.csv from last step in dataset main dir.
        - Go to main directory.
        - Run: python tools/data/basketball3/bb4_bmn_create_labels_file.py --data-prefix data/basketball3 --data-info bb4_video_info.csv --labels-file bb4_proposals_gt.json --frame-interval 6


At this point the folder structure is:
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── basketball3
│   │   ├── bb4_proposals_gt.json
│   │   ├── bb4_video_info.csv
│   │   ├── bb4_bmn_anno_train.json
│   │   ├── bb4_bmn_anno_test.json
│   │   ├── bb4_bmn_anno_validation.json
│   │   ├── rgb_feat
│   │   ├── flow_feat
│   │   ├── extended_clips
│   │   ├── rawframes
│   │   ├── processed_rgb_n_flow_feat
│   │   |   ├── train
│   │   │   │   ├── fb1_u2qAx-8ACOA.csv
│   │   │   │   :
│   │   |   ├── test
│   │   │   │   ├── fb1_6WPSYIdbhF8.csv
│   │   │   │   :
│   │   |   ├── validation
│   │   │   │   ├── fb1_vci_9DF3VDU.csv
│   │   │   │   :

8. Train BMN:
    - Put bmn_400x100_9e_activitynet_feature_20200619-42a3b111.pth in the checkpoints directory (pre-trained weights).
    - Put bb4_bmn_fi6_config.py in the configs/localization/bmn/ directory.
    - Verify that bb4_bmn_anno_train.json and bb4_bmn_anno_test.json are in the basketball3 data directory.
    - Run: python tools/train.py configs/localization/bmn/bb4_bmn_fi6_config.py --work-dir work_dirs/bb4_bmn_fi6 --validate

9. Test BMN (Optional) TODO!

10. Use .pth weights file from work_dirs/bb4_bmn_fi6/ directory for inference (See bb4_bmn_multi_inference.md).