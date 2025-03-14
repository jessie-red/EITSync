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




class JointSender:
    def __init__(self, server_address='127.0.0.1', server_port=5005):
        """Initialize the socket for sending joint data."""
        self.server_address = server_address
        self.server_port = server_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_joints(self, rarm_joints):
        """Send rarm_joints to the visualization script."""
        message = pickle.dumps(rarm_joints)
        self.sock.sendto(message, (self.server_address, self.server_port))

    def close_socket(self):
        """Close the socket."""
        self.sock.close()


class IMUSet:
    """
    Sensor order: left forearm, right forearm, left lower leg, right lower leg, head, pelvis
    """
    def __init__(self, imu_host='127.0.0.1', imu_port=7777, buffer_len=26):
        """
        Init an IMUSet for Noitom Perception Legacy IMUs. Please follow the instructions below.

        Instructions:
        --------
        1. Start `Axis Legacy` (Noitom software).
        2. Click `File` -> `Settings` -> `Broadcasting`, check `TCP` and `Calculation`. Set `Port` to 7002.
        3. Click `File` -> `Settings` -> `Output Format`, change `Calculation Data` to
           `Block type = String, Quaternion = Global, Acceleration = Sensor local`
        4. Place 1 - 6 IMU on left lower arm, right lower arm, left lower leg, right lower leg, head, root.
        5. Connect 1 - 6 IMU to `Axis Legacy` and continue.

        :param imu_host: The host that `Axis Legacy` runs on.
        :param imu_port: The port that `Axis Legacy` runs on.
        :param buffer_len: Max number of frames in the readonly buffer.
        """
        self.imu_host = imu_host
        self.imu_port = imu_port
        self.clock = Clock()

        self._imu_socket = None
        self._buffer_len = buffer_len
        self._quat_buffer = []
        self._acc_buffer = []
        self._is_reading = False
        self._read_thread = None

    def _read(self):
        """
        The thread that reads imu measurements into the buffer. It is a producer for the buffer.
        """
        while self._is_reading:
            data, addr = self._imu_socket.recvfrom(1024)
            data_str = data.decode("utf-8")

            a = np.array(data_str.split("#")[0].split(",")).astype(np.float64)
            q = np.array(data_str.split("#")[1].strip("$").split(",")).astype(np.float64)

            acc = a.reshape((-1, 3))
            quat = q.reshape((-1, 4))

            tranc = int(len(self._quat_buffer) == self._buffer_len)
            self._quat_buffer = self._quat_buffer[tranc:] + [quat.astype(float)]
            self._acc_buffer = self._acc_buffer[tranc:] + [-9.8 * acc.astype(float)]
            self.clock.tick()

    def start_reading(self):
        """
        Start reading imu measurements into the buffer.
        """
        if self._read_thread is None:
            self._is_reading = True
            self._read_thread = threading.Thread(target=self._read)
            self._read_thread.setDaemon(True)
            self._quat_buffer = []
            self._acc_buffer = []
            self._imu_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._imu_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._imu_socket.bind((self.imu_host, self.imu_port))
            self._read_thread.start()
        else:
            print('Failed to start reading thread: reading is already start.')

    def stop_reading(self):
        """
        Stop reading imu measurements.
        """
        if self._read_thread is not None:
            self._is_reading = False
            self._read_thread.join()
            self._read_thread = None
            self._imu_socket.close()

    def get_current_buffer(self):
        """
        Get a view of current buffer.

        :return: Quaternion and acceleration torch.Tensor in shape [buffer_len, 6, 4] and [buffer_len, 6, 3].
        """
        q = torch.from_numpy(np.array(self._quat_buffer)).float()
        a = torch.from_numpy(np.array(self._acc_buffer)).float()
        return q, a

    def get_mean_measurement_of_n_second(self, num_seconds=3, buffer_len=120):
        """
        Start reading for `num_seconds` seconds and then close the connection. The average of the last
        `buffer_len` frames of the measured quaternions and accelerations are returned.
        Note that this function is blocking.

        :param num_seconds: How many seconds to read.
        :param buffer_len: Buffer length. Must be smaller than 60 * `num_seconds`.
        :return: The mean quaternion and acceleration torch.Tensor in shape [6, 4] and [6, 3] respectively.
        """
        save_buffer_len = self._buffer_len
        self._buffer_len = buffer_len
        self.start_reading()
        time.sleep(num_seconds)
        self.stop_reading()
        q, a = self.get_current_buffer()
        self._buffer_len = save_buffer_len
        return q.mean(dim=0), a.mean(dim=0)





class IMUManager:

    IMU_DATA_DICT_ENTRY = "imu_data"
    ARM_DATA_DICT_ENTRY = "arm_data"
    CALIBRATION_DATA_DICT_ENTRY  = "calibration"

    def __init__(self, data_dict, imu_data_dict_entry = IMU_DATA_DICT_ENTRY, arm_data_dict_entry = ARM_DATA_DICT_ENTRY, calibration_data_dict_entry = CALIBRATION_DATA_DICT_ENTRY,
    vis = False, use_phone_as_watch = False, use_right_watch = True, combo = 'rw_rp'):
        self.collect_data = False
        self.data_dict = data_dict
        self.imu_data_dict_entry = imu_data_dict_entry
        self.arm_data_dict_entry = arm_data_dict_entry
        self.calibration_data_dict_entry = calibration_data_dict_entry
        self.imu_data = (0,0)
        self.arm_data = np.zeros((5, 3))
        self.vis = vis     
        self.running = False   
        # Configurations 
        self.use_phone_as_watch = use_phone_as_watch
        self.use_right_watch = use_right_watch
        self.right_arm = [17, 19, 21, 23]
        self.left_arm = [16, 18, 20, 22] 
        self.combo = combo


    def run(self, imudone): 


        # specify device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # setup IMU collection
        imu_set = IMUSet(buffer_len=1)

        # align IMU to SMPL body frame
        input('Put imu 1 aligned with your body reference frame (x = Left, y = Up, z = Forward) and then press any key.')
        print('Keep for 3 seconds ...', end='')
        oris = imu_set.get_mean_measurement_of_n_second(num_seconds=3, buffer_len=40)[0][0]
        smpl2imu = quaternion_to_rotation_matrix(oris).view(3, 3).t()

        # bone and acceleration offset calibration
        input('\tFinish.\nWear all imus correctly and press any key.')
        for i in range(3, 0, -1):
            print('\rStand straight in T-pose and be ready. The calibration will begin after %d seconds.' % i, end='')
            time.sleep(1)
        print('\nStand straight in T-pose. Keep the pose for 3 seconds ...', end='')

        oris, accs = imu_set.get_mean_measurement_of_n_second(num_seconds=3, buffer_len=40)
        oris = quaternion_to_rotation_matrix(oris)
        device2bone = smpl2imu.matmul(oris).transpose(1, 2).matmul(torch.eye(3))
        acc_offsets = smpl2imu.matmul(accs.unsqueeze(-1))   # [num_imus, 3, 1], already in global inertial frame

        # start streaming data
        print('\tFinished Calibrating.\nEstimating poses. Press q to quit')
        imu_set.start_reading()

        # load model
        model = load_model(paths.weights_file)

        # setup Unity server for visualization
        if self.vis:
            server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_for_unity.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            server_for_unity.bind(('0.0.0.0', 8889))
            server_for_unity.listen(1)
            print('Server start. Waiting for unity3d to connect.')
            conn, addr = server_for_unity.accept()

        self.running = True
        clock = Clock()
        record_buffer = None

        data_dict[self.calibration_data_dict_entry] = (smpl2imu, device2bone)


        n_imus = 5
        accs, oris = [], []
        raw_accs, raw_oris = [], []
        poses, trans = [], []

        imudone.set()

        model.eval()
        while self.running:
            # calibration
            clock.tick(datasets.fps)
            ori_raw, acc_raw = imu_set.get_current_buffer() # [buffer_len, 5, 4]
            ori_raw = quaternion_to_rotation_matrix(ori_raw).view(-1, n_imus, 3, 3)
            glb_acc = (smpl2imu.matmul(acc_raw.view(-1, n_imus, 3, 1)) - acc_offsets).view(-1, n_imus, 3)
            glb_ori = smpl2imu.matmul(ori_raw).matmul(device2bone)

            # normalization 
            _acc = glb_acc.view(-1, 5, 3)[:, [1, 4, 3, 0, 2]] / amass.acc_scale
            _ori = glb_ori.view(-1, 5, 3, 3)[:, [1, 4, 3, 0, 2]]
            acc = torch.zeros_like(_acc)
            ori = torch.zeros_like(_ori)

            # device combo
            c = amass.combos[self.combo]

            if self.use_phone_as_watch:
                # set watch value to phone
                acc[:, [1]] = _acc[:, [3]]
                ori[:, [1]] = _ori[:, [3]]
            elif self.use_right_watch:
                acc[:, [1]] = _acc[:, [0]]
                ori[:, [1]] = _ori[:, [0]]
            else:
                # filter and concat input
                acc[:, c] = _acc[:, c] 
                ori[:, c] = _ori[:, c]
            
            imu_input = torch.cat([acc.flatten(1), ori.flatten(1)], dim=1)
            self.imu_data = (acc[:,[1]].flatten(), ori[:,[1]].flatten())

            if self.collect_data:
                linux_time = time.time()
                if self.imu_data_dict_entry not in self.data_dict.keys():
                    self.data_dict[self.imu_data_dict_entry] = [(linux_time, acc[:,[1]], ori[:,[1]], acc_raw, ori_raw)]
                else:
                    self.data_dict[self.imu_data_dict_entry].append([(linux_time, acc[:,[1]], ori[:,[1]])])

            # predict pose and translation
            with torch.no_grad():
                output = model.forward_online(imu_input.squeeze(0), [imu_input.shape[0]])
                pred_pose = output[0] # [24, 9]
                pred_joint = output[1] # [24, 3]
                pred_tran = output[2] # [3]

            # freeze left arm pose
            for j in self.left_arm:
                pred_pose[j] = torch.eye(3).view(9)

            # send right arm joints to visualizer
            rarm_joints = pred_joint[self.right_arm].cpu().numpy()  # Extract right arm joint positions
            ##TODO: add joints to data_dict
            #joint_sender.send_joints(rarm_joints)
            self.arm_data = rarm_joints
            if self.collect_data:
                linux_time = time.time()
                if self.arm_data_dict_entry not in self.data_dict.keys():
                    self.data_dict[self.arm_data_dict_entry] = [(linux_time, rarm_joints)]
                else:
                    self.data_dict[self.arm_data_dict_entry].append([(linux_time, rarm_joints)])


            # send pose
            if self.vis:
                s = ','.join(['%g' % v for v in pose]) + '#' + \
                    ','.join(['%g' % v for v in tran]) + '$'
                conn.send(s.encode('utf8'))  
                
                if os.getenv("DEBUG") is not None:
                    print('\r', '(recording)' if is_recording else '', 'Sensor FPS:', imu_set.clock.get_fps(),
                            '\tOutput FPS:', clock.get_fps(), end='')


        # clean up threads
        get_input_thread.join()
        imu_set.stop_reading()
        joint_sender.close_socket()
        print('Finish.')

    def start_data_collection(self):
        self.collect_data = True

    def stop_data_collection(self):
        self.collect_data = False

    def get_recent_data_imu(self):
        return self.imu_data

    def get_recent_data_arm(self):
        return self.arm_data
