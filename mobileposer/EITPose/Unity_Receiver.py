"""
Code adapted from: https://github.com/Xinyu-Yi/TransPose/blob/main/live_demo.py
"""

import os
import sys
import time
import socket
import threading
import torch
import numpy as np
from datetime import datetime
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from pygame.time import Clock
import pickle
import subprocess

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication
from pyqtgraph.Qt import QtCore

from articulate.math import *
from mobileposer.models import *
from mobileposer.utils.model_utils import *
from mobileposer.config import *

"""
The gesture index to gesture correspondance is as follows:
    0: claw
    1: spiderman
    2: hang ten
    3: OK
    4: point
    5: L
    6: thumbs up
    7: open hand
    8: fist
    9: two finger point
    10: middle finger thumb touch
    11: fist with thumb to side

The orientation index is (direction of palm):
    0: Down
    1: Left
    2: Right
    3: Up
"""








class UnityManager:

    HAND_DATA_DICT_ENTRY = "hand_data"
    HEAD_DATA_DICT_ENTRY = "head_data"
    GESTURE_DATA_DICT_ENTRY = "gesture_data"
    BODY_DATA_DICT_ENTRY = "body_data"

    def __init__(self, data_dict, hand_data_dict_entry = HAND_DATA_DICT_ENTRY, head_data_dict_entry = HEAD_DATA_DICT_ENTRY, gesture_data_dict_entry = GESTURE_DATA_DICT_ENTRY, 
    body_data_dict_entry = BODY_DATA_DICT_ENTRY):
        self.collect_data = False
        self.data_dict = data_dict
        self.hand_data_dict_entry = hand_data_dict_entry
        self.head_data_dict_entry = head_data_dict_entry
        self.gesture_data_dict_entry = gesture_data_dict_entry
        self.body_data_dict_entry = body_data_dict_entry
        self.running = True 
        self.joint_rotations = torch.zeros(24,4)
        self.head_pos = []
        self.head_rot = []
        self.hand_pos = []
        self.hand_rot = []
        self.ball_pos = []
        self.gesture = []
        self.orientation = []
        self.body_pos = []
        self.body_rot = []
          
        # Configurations 
        



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
                    #print(f"\r{buffer}")
                    message, buffer = buffer.split('$', 1)
                    
                    self.parse_message(message)
                    pose = rotation_matrix_to_axis_angle(torch.zeros((24,3,3)).view(1, 216)).view(72)
                    #self.send_data(pose, torch.zeros(3), self.joint_rotations)
                    #
                    if self.collect_data:
                        linux_time = time.time()
                        self.record_data(linux_time)
            except Exception as e:
                print(f"Error with data: {e}")
                break
                


    def send_data(self, pose, tran, hand):
        joint_strings = [f"{q[0]}, {q[1]},{q[2]},{q[3]}" for q in hand]
        s = ','.join(['%g' % v for v in pose]) + '#' + \
            ','.join(['%g' % v for v in tran]) + '#' + \
            ';'.join(joint_strings) + '$'
        self.conn.send(s.encode('utf8'))  

    def parse_message(self, data):
        head, hand, pos, body, _ = data.split('!')
        if len(head.split('#')) > 1:
            self.head_pos, self.head_rot = head.split('#')
            self.head_pos = self.head_pos.split(',')
            self.head_rot = self.head_rot.split(',')
        hand_data = hand.split('#')
        if len(hand_data) > 1:
            self.hand_pos = hand_data[0].split(',')
            self.hand_rot = hand_data[1].split(',')
            joint_rotations = hand_data[2:]
            self.joint_rotations = []
            for joint in joint_rotations:
                x,y,z,w = map(float, joint.split(','))
                self.joint_rotations.append((x,y,z,w))
        ball_data = pos.split('#')
        #print(f"\r{ball_data}")
        if len(ball_data) > 1:
            self.ball_pos = ball_data[0].split(',')
            self.gesture = ball_data[1]
            self.orientation = ball_data[2]
        body_joints = body.split('#')
        self.body_pos = []
        self.body_rot = []
        if len(body_joints) > 1:
            #print("yes body!")
            for idx in range(0,len(body_joints), 2):
                self.body_pos.append(body_joints[idx].split(','))
                self.body_rot.append(body_joints[idx+1].split(','))
        #else:
            #print("no body :(")


        



    def get_recent_joint_data(self):
        return self.joint_rotations

    def get_recent_hand_data(self):
        return self.hand_pos, self.hand_rot

    def get_recent_head_data(self):
        return self.head_pos, self.head_rot

    def get_recent_gesture_data(self):
        return self.ball_pos, self.gesture, self.orientation
    def get_recent_body_data(self):
        return self.body_pos, self.body_rot

    def record_data(self, linux_time):
        if self.hand_data_dict_entry not in self.data_dict.keys():
            self.data_dict[self.hand_data_dict_entry] = [(linux_time, self.hand_pos, self.hand_rot, self.joint_rotations)]
        else:
            self.data_dict[self.hand_data_dict_entry].append((linux_time, self.hand_pos, self.hand_rot, self.joint_rotations))
        if self.head_data_dict_entry not in self.data_dict.keys():
            self.data_dict[self.head_data_dict_entry] = [(linux_time, self.head_pos, self.head_rot)]
        else:
            self.data_dict[self.head_data_dict_entry].append((linux_time, self.head_pos, self.head_rot))
        if self.gesture_data_dict_entry not in self.data_dict.keys():
            self.data_dict[self.gesture_data_dict_entry] = [(linux_time, self.ball_pos, self.gesture, self.orientation)]
        else:
            self.data_dict[self.gesture_data_dict_entry].append((linux_time, self.ball_pos, self.gesture, self.orientation))
        if self.body_data_dict_entry not in self.data_dict.keys():
            self.data_dict[self.body_data_dict_entry] = [(linux_time, self.body_pos, self.body_rot)]
        else:
            self.data_dict[self.body_data_dict_entry].append((linux_time, self.body_pos, self.body_rot))
