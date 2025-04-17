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
class UnityManager:

    def __init__(self):
        self.running = True

        



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



username = input("Please enter username: (the most recent dataset with that name will have hand poses found)")
#username = "Video"
pattern = os.path.join(paths.processed_data, f"{username}.pkl")
files = glob.glob(pattern)
#print(files)
if not files:
    print("no files found for", username)
else:
    df = pd.read_pickle(files[0])
    print("dataframe found!")
#print(df.shape[0])
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
print("writing")
df.to_pickle(paths.processed_data / f"{username}_pos.pkl")
df.to_csv(paths.processed_data / f"{username}_pos.csv")