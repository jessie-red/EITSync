import sys
from PyQt5.QtWidgets import QApplication
from threading import Condition, Thread, Event
import pandas as pd
import datetime
import time
import os
import socket
import subprocess
from pathlib import Path

from mobileposer.config import *
from EITPose.RecorderVisualizer import MainWindow
from EITPose.TeensySerial_Receiver import TeensySerialManager
from EITPose.IMU_Receiver import IMUManager
from EITPose.Unity_Receiver import UnityManager


if __name__ == '__main__':
    data_dict = {}
    print('Starting Data Recorder...')


    # # Start Bluetooth Connection
    # teensybt = TeensyBluetoothManager(BLUETOOTH_ADDRESS, SERVICE_UUID, CHARACTERISTIC_UUID, data_dict=data_dict)
    # teensybt_thread = Thread(target=teensybt.run, daemon=True)
    # teensybt_thread.start()

    #setting up the server for updates from the watch

    imudone = Event()
    unityconnected = Event()

    teensybt = TeensySerialManager(data_dict, numElect=8, serialPortName=TeensySerialManager.PORT_NAME)
    teensybt_thread = Thread(target=teensybt.run, daemon=True)
    teensybt_thread.start()

    

    calibration = subprocess.Popen(["python", paths.calibration_file], stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT)

    mobileimu = IMUManager(data_dict)
    mobileimu_thread = Thread(target=mobileimu.run, daemon=True, args = (imudone,))
    mobileimu_thread.start()

    imudone.wait()
    print("IMUs Calibrated")
    print("Put on your headset!")

    unity = UnityManager(data_dict)
    unity_thread = Thread(target=unity.run, daemon=True, args = (unityconnected,))
    unity_thread.start()

    unityconnected.wait()
    print("Unity Connected")

   

    # test data for testing the visualizer without bluetooth device nearby
    # data_dict.update({TeensyBluetoothConnectionThread.DATA_DICT_ENTRY: [(time.time(), [0]*64 + [ 9.81, 0, 0, 356, 120, -60, 3, 4, 5, -0.847, -0.489, -0.197, 0.066])]})


    def clear_data_dict():
        data_dict.clear()

    def save_data_dict(gesture_name, checkpoint = False):
        # Get the current date and time
        now = datetime.datetime.now()
        # Convert the date and time to a string in the format YYYYMMDD_HHMMSS
        save_time = now.strftime("%Y%m%d_%H%M%S")

        # Convert each list in the dictionary into a DataFrame and save it as a .pkl file
        for key, value in data_dict.items():
            if key is TeensySerialManager.DATA_DICT_ENTRY:
                df = pd.DataFrame(value, columns=['timestamp', 'data'])
            elif key is TeensySerialManager.DATA_DICT_ENTRY_EIT:
                df = pd.DataFrame(value, columns=['timestamp', 'eit_data'])
            elif key is IMUManager.IMU_DATA_DICT_ENTRY:
                df = pd.DataFrame(value, columns=['timestamp', 'acc', 'ori', 'acc_raw', 'ori_raw'])
            elif key is IMUManager.POSE_DATA_DICT_ENTRY:
                df = pd.DataFrame(value, columns=['timestamp', 'arm_data', 'pose', 'tran'])
            elif key is IMUManager.CALIBRATION_DATA_DICT_ENTRY:
                df = pd.DataFrame(value, columns = ['smpl2imu','device2bone'])
            elif key is UnityManager.HAND_DATA_DICT_ENTRY:
                df = pd.DataFrame(value, columns = ['timestamp','hand_pos', 'hand_rot', 'joint_data'])
            elif key is UnityManager.HEAD_DATA_DICT_ENTRY:
                df = pd.DataFrame(value, columns = ['timestamp','head_pos', 'head_rot'])
            elif key is UnityManager.GESTURE_DATA_DICT_ENTRY:
                df = pd.DataFrame(value, columns = ['timestamp','ball_pos', 'gesture', 'orientation'])
            elif key is UnityManager.BODY_DATA_DICT_ENTRY:
                df = pd.DataFrame(value, columns = ['timestamp','body_pos', 'body_rot'])
            elif key == 'gesture':
                df = pd.DataFrame(value, columns=['timestamp', 'round', 'gesture'])
            if checkpoint:
                df.to_pickle(paths.raw_data / f'checkpoint_{key}.pkl')
            else:
                df.to_pickle(paths.raw_data / f'{gesture_name}_{key}_{save_time}.pkl')

            #df.to_pickle(f'data/{gesture_name}_{key}_{save_time}.pkl')
    


    print("Starting GUI...")
    app = QApplication(sys.argv)
    style = 'dark'
    if style == 'dark':
        app.setStyleSheet("""
        QWidget { 
            background-color: #0B0B0B; 
        }
        QPushButton {
            color: white;
            background-color: gray;
            border: 1px solid white;
            border-radius: 10px;   /* rounded corners */
            padding: 5px;
            font: 14px "Helvetica";
        }
        QPushButton:hover {
            background-color: darkgray;
        }
        QPushButton:checked {
            background-color: red;
        }
        QPushButton:pressed {
            background-color: darkred;
        }
        QLabel {
            color: white;
        }
        QLineEdit {
            color: white;
        }

        """)
    else:
        app.setStyleSheet("""
        QWidget { 
            background-color: #FBFBFB; 
        }
        QPushButton {
            color: black;
            background-color: lightgray;
            border: 1px solid black;
            border-radius: 10px;   /* rounded corners */
            padding: 5px;
            font: 14px "Helvetica";
        }
        QPushButton:hover {
            background-color: darkgray;
        }
        QPushButton:checked {
            background-color: red;
        }
        QPushButton:pressed {
            background-color: darkred;
        }
        QLabel {
            color: black;
        }
        """)


    window = MainWindow(data_dict, save_data_dict, clear_data_dict, teensybt, mobileimu, unity)
    window.show()
    sys.exit(app.exec_())
