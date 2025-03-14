import sys
from PyQt5.QtWidgets import QApplication
from threading import Condition, Thread, Event
import pandas as pd
import datetime
import time
import os
import socket
import subprocess

from mobileposer.config import *
from EITPose.RecorderVisualizer import MainWindow
from EITPose.TeensySerial_Receiver import TeensySerialManager
from EITPose.IMU_Receiver import IMUManager


if __name__ == '__main__':
    data_dict = {}
    print('Starting Data Recorder...')


    # # Start Bluetooth Connection
    # teensybt = TeensyBluetoothManager(BLUETOOTH_ADDRESS, SERVICE_UUID, CHARACTERISTIC_UUID, data_dict=data_dict)
    # teensybt_thread = Thread(target=teensybt.run, daemon=True)
    # teensybt_thread.start()

    #setting up the server for updates from the watch

    imudone = Event()

    teensybt = TeensySerialManager(data_dict, numElect=8, serialPortName=TeensySerialManager.PORT_NAME)
    teensybt_thread = Thread(target=teensybt.run, daemon=True)
    teensybt_thread.start()

    calibration = subprocess.Popen(["python", paths.calibration_file])

    mobileimu = IMUManager(data_dict)
    mobileimu_thread = Thread(target=mobileimu.run, daemon=True, args = (imudone,))
    mobileimu_thread.start()

    imudone.wait()

    # test data for testing the visualizer without bluetooth device nearby
    # data_dict.update({TeensyBluetoothConnectionThread.DATA_DICT_ENTRY: [(time.time(), [0]*64 + [ 9.81, 0, 0, 356, 120, -60, 3, 4, 5, -0.847, -0.489, -0.197, 0.066])]})

        
    def start_data_collection():
        teensybt.start_data_collection()
        mobileimu.start_data_collection()


    def stop_data_collection():
        teensybt.stop_data_collection()
        mobileimu.stop_data_collection()

    def clear_data_dict():
        data_dict.clear()

    def save_data_dict(gesture_name):
        # Get the current date and time
        now = datetime.datetime.now()
        # Convert the date and time to a string in the format YYYYMMDD_HHMMSS
        save_time = now.strftime("%Y%m%d_%H%M%S")

        # Convert each list in the dictionary into a DataFrame and save it as a .pkl file
        for key, value in data_dict.items():
            if key is TeensySerialManager.DATA_DICT_ENTRY:
                df = pd.DataFrame(value, columns=['timestamp', 'data'])
            elif key is TeensySerialManager.DATA_DICT_ENTRY_EIT:
                df = pd.DataFrame(value, columns=['timestamp', 'data'])
            elif key is IMUManager.IMU_DATA_DICT_ENTRY:
                df = pd.DataFrame(value, columns=['timestamp', 'acc', 'ori', 'acc_raw', 'ori_raw'])
            elif key is IMUManager.ARM_DATA_DICT_ENTRY:
                df = pd.DataFrame(value, columns=['timestamp', 'data'])
            elif key is IMUManager.CALIBRATION_DATA_DICT_ENTRY:
                df = pd.DataFrame(value, columns = ['smpl2imu','device2bone'])
            elif key == 'gesture':
                df = pd.DataFrame(value, columns=['timestamp', 'round', 'gesture'])
            df.to_pickle(f'/data/{gesture_name}_{key}_{save_time}.pkl')
    

    time.sleep(5) # sleep for 10 seconds to allow threads to start

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


    window = MainWindow(data_dict, start_data_collection, stop_data_collection, save_data_dict, clear_data_dict, teensybt, mobileimu)
    window.show()
    sys.exit(app.exec_())
