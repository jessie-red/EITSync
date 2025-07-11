import torch
from pathlib import Path
from enum import Enum, auto


class train_hypers:
    """Training hyperparameters."""
    batch_size = 256
    num_workers = 8
    num_epochs = 100
    accelerator = "gpu"
    device = "0"
    lr = 1e-3


class paths:
    """Relevant paths for MobilePoser. Change as necessary."""
    root_dir = Path().absolute() / "mobileposer"
    smpl_file = root_dir / 'smpl/basicmodel_m.pkl'
    #weights_file = root_dir / 'checkpoints/model_finetuned.pth'
    weights_file = root_dir / "weights.pth"
    calibration_file = root_dir / "calibration.py"
    mesh_data = root_dir/ "armdata.mat"
    
    raw_data = root_dir / "raw_data"
    processed_data = root_dir / "processed_data"
    processed_datasets25 = root_dir / "data/processed_25fps"
    processed_dev = root_dir / "data/processed_dev"       # subset for fast testing
    dev_data = root_dir / "data/dev_dataset" # stores data for development testing 


class model_config:
    """MobilePoser Model configurations."""
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # joint set
    n_joints = 5                        # (head, right-wrist, left-wrist, right-hip, left-hip)
    n_imu = 12*n_joints                 # 60 (3 accel. axes + 3x3 orientation rotation matrix) * 5 possible IMU locations
    n_output_joints = 24                # 24 output joints
    n_pose_output = n_output_joints*6   # 144 pose output (24 output joints * 6D rotation matrix)

    # model config
    past_frames = 40
    future_frames = 5
    total_frames = past_frames + future_frames


class amass:
    """AMASS dataset information."""
    # device-location combinationsa
    combos = {
        # 'lw_rp_h': [0, 3, 4],
        # 'rw_rp_h': [1, 3, 4],
        # 'lw_lp_h': [0, 2, 4],
        # 'rw_lp_h': [1, 2, 4],
        # 'lw_lp': [0, 2],
        'lw_rp': [0, 3],
        'rw_lp': [1, 2],
        'rw_rp': [1, 3],
        # 'lp_h': [2, 4],
        # 'rp_h': [3, 4],
        # 'lp': [2],
        # 'rp': [3],
        'rw' : [1]
     }
    acc_scale = 30
    vel_scale = 2

    # left wrist, right wrist, left thigh, right thigh, head, pelvis
    all_imu_ids = [0, 1, 2, 3, 4] 
    imu_ids = [0, 1, 2, 3]

    pred_joints_set = [*range(24)]
    joint_sets = [18, 19, 1, 2, 15, 0]
    ignored_joints = list(set(pred_joints_set) - set(joint_sets))


class datasets:
    """Dataset information."""
    # FPS of data
    fps = 25

    # DIP dataset
    dip_name = "dip"
    dip_test = "dip_test.pt"
    dip_train = "dip_train.pt"

    # TotalCapture dataset
    totalcapture = "TotalCapture.pt"
    totalcapture_all = f"totalcapture{fps}.pt"
    totalcapture_test = "totalcapture_test.pt"
    totalcapture_train = "totalcapture_train.pt"

    # IMUPoser dataset
    imuposer_all = "imuposer.pt"
    imuposer_train = "imuposer_train.pt"
    imuposer_test = "imuposer_test.pt"

    # Datasets to ignore during training
    train_ignore = [totalcapture_all, totalcapture_test, totalcapture_train,
                    dip_test, dip_train, imuposer_all]

    # Test datasets
    test_datasets = {
        'dip': dip_test,
        'totalcapture': totalcapture_all,
        'imuposer': imuposer_test
    }

    # Finetune datasets
    finetune_datasets = {
        'dip': dip_train,
        'imuposer': imuposer_train
    }

    # AMASS datasets
    amass_datasets = ['ACCAD', 'BioMotionLab_NTroje', 'BMLhandball', 'BMLmovi', 'CMU',
                      'DanceDB', 'DFaust_67', 'EKUT', 'Eyes_Japan_Dataset', 'HUMAN4D',
                      'HumanEva', 'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU',
                      'SSM_synced', 'TCD_handMocap', 'TotalCapture', 'Transitions_mocap']

    # Root-relative joint positions
    root_relative = False

    # Window length of IMU and Pose data 
    window_length = 125


class joint_set:
    """Joint sets configurations."""
    gravity_velocity = -0.018

    full = list(range(0, 24))
    reduced = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    n_full = len(full)
    n_ignored = len(ignored)
    n_reduced = len(reduced)

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]


class sensor: 
    """Sensor parameters."""
    device_ids = {
        'Left_phone': 0,
        'Left_watch': 1,
        'Left_headphone': 2,
        'Right_phone': 3,
        'Right_watch': 4
    }


class Devices(Enum):
    """Device IDs."""
    Left_Phone = auto()
    Left_Watch = auto()
    Right_Headphone = auto()
    Right_Phone = auto()
    Right_Watch = auto()
