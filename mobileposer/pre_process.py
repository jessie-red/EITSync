import pandas as pd
from mobileposer.config import *
from pathlib import Path
import glob
import os
from datetime import datetime
import numpy as np

def get_latest_files_for_user(username):
    # Find all files matching the pattern
    pattern = os.path.join(paths.raw_data, f"{username}*.pkl")
    files = glob.glob(pattern)
    #print(pattern)
    
    if not files:
        print("no files found for", username)
        return []
    
    # Extract dates from filenames
    date_dict = {}
    for file in files:
        # Extract date part (assumes format is username_data_date.pkl)
        #print(file.split('_data_'))
        date_str = file.split('_data_')[1].replace('.pkl', '')
        
        try:
            # Convert to datetime object for proper comparison
            date_obj = datetime.strptime(date_str, "%Y%m%d_%H%M%S")  # Adjust format as needed
            
            if date_obj not in date_dict:
                date_dict[date_obj] = []
            
            date_dict[date_obj].append(file)
        except ValueError:
            # Skip files that don't match the expected date format
            continue
    
    if not date_dict:
        return []
    
    # Find the most recent date
    latest_date = max(date_dict.keys())
    
    # Return all files from that date
    return date_dict[latest_date]


def efficient_merge_by_timestamp(main_df, other_dfs):
    # Ensure timestamp columns are datetime
    main_df = main_df.copy()
    #main_df['timestamp'] = pd.to_datetime(main_df['timestamp'])
    
    for i, df in enumerate(other_dfs):
        df_copy = df.copy()
        
        # Add suffix to avoid column conflicts
        rename_dict = {col: f"{col}" for col in df_copy.columns if col != 'timestamp'}
        df_copy = df_copy.rename(columns=rename_dict)
        
        # Merge using merge_asof which finds the nearest match before each timestamp
        main_df = pd.merge_asof(
            main_df.sort_values('timestamp'), 
            df_copy.sort_values('timestamp'),
            on='timestamp', 
            direction='backward'  # Get the most recent value before or equal to the timestamp
        )
    
    return main_df

def process_ori(item):
    # Check if item is a tensor
    if isinstance(item, torch.Tensor):
        # Check if it has the expected shape dimensions
        if len(item.shape) == 4 and item.shape[1] >= 2:
            # Extract item[:,1,:] - this gets all batches, the 2nd channel, all rows and columns
            return item[:,1,:,:].squeeze()
        else:
            # If tensor doesn't have expected shape, return it unchanged
            return item
    else:
        # If not a tensor (e.g., a list), return unchanged
        return np.zeros((3,3))
def process_acc(item):
    # Check if item is a tensor
    if isinstance(item, torch.Tensor):
        # Check if it has the expected shape dimensions
        if len(item.shape) == 3 and item.shape[1] >= 2:
            # Extract item[:,1,:] - this gets all batches, the 2nd channel, all rows and columns
            return item[:,1,:].squeeze()
        else:
            # If tensor doesn't have expected shape, return it unchanged
            return item
    else:
        # If not a tensor (e.g., a list), return unchanged
        return [0,0,0]
def get_eit_avg_vals(eit_data):
        return np.array([sum(eit_data[0+i:64:8])/8 for i in range(8)])
    
def remove_eit_zeros(eit_data):
    return np.array([eit_dat for eit_dat in eit_data if eit_dat != 0.0])

username = input("Please enter username: (the most recent dataset with that name will be processed)")
#username = "test"
latest_files = get_latest_files_for_user(username)
#print(f"Latest files for {username}: {latest_files}")
body_df, eit_df, gesture_df, hand_df, head_df, imu_df, pose_df = [[] for _ in range(7)]
for filename in latest_files:
    file = filename.split("/")[-1].split("_")[1]
    #print(file)
    if "body" in file:
        body_df = pd.read_pickle(filename)
        body_df['body_pos'] = body_df['body_pos'].apply(lambda x: [[float(item) for item in sublist] for sublist in x])
        body_df['body_rot'] = body_df['body_rot'].apply(lambda x: [[float(item) for item in sublist] for sublist in x])
    elif "eit" in file:
        eit_df = pd.read_pickle(filename)
        eit_df['eit_data'] = eit_df['eit_data'].apply(lambda x: np.hstack((remove_eit_zeros(x), get_eit_avg_vals(x))))
    elif "gesture" in file:
        gesture_df = pd.read_pickle(filename)
        gesture_df['ball_pos'] = gesture_df['ball_pos'].apply(lambda x: [float(item) for item in x])
    elif "hand" in file:
        hand_df = pd.read_pickle(filename)
        hand_df['hand_pos'] = hand_df['hand_pos'].apply(lambda x: [float(item) for item in x])
        hand_df['hand_rot'] = hand_df['hand_rot'].apply(lambda x: [float(item) for item in x])
    elif "head" in file:
        head_df = pd.read_pickle(filename)
        head_df['head_pos'] = head_df['head_pos'].apply(lambda x: [float(item) for item in x])
        head_df['head_rot'] = head_df['head_rot'].apply(lambda x: [float(item) for item in x])
    elif "imu" in file:
        imu_df = pd.read_pickle(filename)
        imu_df['ori'] = imu_df['ori'].apply(process_ori)
        imu_df['acc'] = imu_df['acc'].apply(process_acc)
        imu_df['ori_raw'] = imu_df['ori_raw'].apply(process_ori)
        imu_df['acc_raw'] = imu_df['acc_raw'].apply(process_acc)
    elif "pose" in file:
        pose_df = pd.read_pickle(filename)


for idx, file  in enumerate([body_df, eit_df, gesture_df, hand_df, head_df, imu_df, pose_df]):
    if len(file) == 0:
        print(f"Missing file, idx {idx}")
#okay now we have to process them into a single dataframe
processed_df = efficient_merge_by_timestamp(imu_df, [body_df, eit_df, gesture_df, hand_df, head_df, pose_df])
#print(processed_df.columns)
processed_df.to_pickle(paths.processed_data / f"{username}.pkl")
processed_df.to_csv(paths.processed_data / f"{username}.csv")


