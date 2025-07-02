import pandas as pd
from mobileposer.config import *
from pathlib import Path
import glob
import os
from datetime import datetime
import numpy as np
from threading import Condition, Thread, Event
import socket
import time
from threading import Condition, Thread, Event
import socket
import time

class UnityManager:
    
    def __init__(self, joint_list, length, endevent):
        self.joint_list = joint_list
        self.running = True
        self.endevent = endevent
        self.length = length

        



    def run(self, unityconnected): 
        server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_for_unity.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        server_for_unity.bind(('0.0.0.0', 8889))
        server_for_unity.listen(1)
        print('Server start. Waiting for unity3d to connect.')
        self.conn, addr = server_for_unity.accept()
        print('Unity server accepted')
        unityconnected.set()
        buffer = ""
        while self.running and self.conn:
            try:
                data = self.conn.recv(4096).decode('utf-8')
                if not data:
                    print("Client disconnected")
                    self.running = False
                    break
                    
                # Add to buffer
                buffer += data
                
                # Process complete messages
                while '$' in buffer:
                    # Extract message until newline
                    #print(f"\rrecd\n", end="")
                    message, buffer = buffer.split('$', 1)
                    
                    self.parse_message(message)

            except Exception as e:
                print(f"Error with data: {e}")
                break
                


    def send_data(self, joint_data):
        joint_strings = [f"{q[0]}, {q[1]},{q[2]},{q[3]}" for q in joint_data]
        joint_strings.insert(0, f"0, 0, 0")
        joint_strings.insert(1, f"0, 0, 0, 0")
        s = '#'.join(joint_strings) + '$'
        self.conn.send(s.encode('utf8'))

    def parse_message(self, data):
        joint_lst = []
        joints = data.split('#')
        for joint in joints:
            if joint != '':
                #print(joint)
                x,y,z = map(float, joint.split(','))
                joint_lst.append((x,y,z))
        self.joint_list.append(joint_lst)
        if len(self.joint_list) == self.length-1:
            endevent.set()
        else:
            length = len(self.joint_list)
            print(f"\rrecieved {length}", end="")



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

def get_fov_from_pos(pos):
    if abs(pos[2]) < .3:
        return 0
    else:
        return 1

def get_fov_from_pos_AR(pos):
    if abs(pos[0]) == .2 and abs(pos[1]) == 0:
        return 0
    else:
        return 1


def process_ball_VR(pos):
    if type(pos) == list:
        return get_fov_from_pos(pos)
    else:
        if pos < 10:
            return 1
        else:
            return 0

def process_ball_AR(pos):
    if type(pos) == list:
        return get_fov_from_pos(pos)
    else:
        if pos < 4:
            return 1
        else:
            return 0
def process_ball(pos):
    if type(pos) == list:
        return [float(item) for item in pos]
    else:
        return float(pos)

username = input("Please enter username: (the most recent dataset with that name will be processed)")
#username = "test"
latest_files = get_latest_files_for_user(username)
#print(f"Latest files for {username}: {latest_files}")
body_df, eit_df, gesture_df, hand_df, head_df, imu_df, pose_df = [[] for _ in range(7)]
for filename in latest_files:
    file = filename.split("/")[-1].split("_")[1]
    #print(file)
    if "eit" in file:
        eit_df = pd.read_pickle(filename)
        eit_df['eit_data'] = eit_df['eit_data'].apply(lambda x: np.hstack((remove_eit_zeros(x), get_eit_avg_vals(x))))
    elif "gesture" in file:
        gesture_df = pd.read_pickle(filename)
        gesture_df['ball_pos'] = gesture_df['ball_pos'].apply(process_ball)
        gesture_df['VR_FOV'] = gesture_df['ball_pos'].apply(process_ball_VR)
        gesture_df['AR_FOV'] = gesture_df['ball_pos'].apply(process_ball_AR)
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

missing_body = True
for idx, file  in enumerate([eit_df, gesture_df, hand_df, head_df, imu_df, pose_df]):
    if len(file) == 0:
        if idx == 0:
            missing_body = True
        else:
            print(f"Missing file, idx {idx}")

#okay now we have to process them into a single dataframe
if missing_body:
    processed_df = efficient_merge_by_timestamp(imu_df, [eit_df, gesture_df, hand_df, head_df, pose_df])
else:
    processed_df = efficient_merge_by_timestamp(imu_df, [body_df, eit_df, gesture_df, hand_df, head_df, pose_df])
#print(processed_df.columns)
df = processed_df
def remove_duplicates(df, name):
    prev = None
    group_ids = []
    group_id = -1

    for val in df[name]:
        if prev is None or not np.array_equal(val, prev):
            group_id += 1
        group_ids.append(group_id)
        prev = val

    df['group'] = group_ids
    df['within_group_idx'] = df.groupby('group').cumcount()

    # Keep only the first 5 repetitions
    return df[df['within_group_idx'] < 5].drop(columns=['group', 'within_group_idx'])


# Track group changes
filtered_df = remove_duplicates(df, 'eit_data')
filtered_df = remove_duplicates(filtered_df, 'acc')
filtered_df = filtered_df[~filtered_df['acc'].apply(lambda x: all(val == 0 for val in x))]

processed_df = filtered_df
processed_df.to_pickle(paths.processed_data / f"{username}.pkl")
processed_df.to_csv(paths.processed_data / f"{username}.csv")

df = processed_df
df = df[~df['acc'].apply(lambda x: all(val == 0 for val in x))]
joint_list = []
unityconnected = Event()
endevent = Event()
unity = UnityManager(joint_list, df.shape[0], endevent)
unity_thread = Thread(target=unity.run, daemon=True, args = (unityconnected,))
unity_thread.start()
unityconnected.wait()
for iter, row in enumerate(df.itertuples()):
    #print(f"\rsending {iter}\n", end = "")
    unity.send_data(row.joint_data)
endevent.wait()
#time.sleep(2)
#print(joint_list[-1])
df.drop(df.index[-1], inplace=True)
df['joint_pos'] = joint_list
print("\nwriting")
df.to_pickle(paths.processed_data / f"{username}_pos.pkl")
df.to_csv(paths.processed_data / f"{username}_pos.csv")


