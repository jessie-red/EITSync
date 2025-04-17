import sys

import sys
from PyQt5.QtWidgets import QApplication
from threading import Condition, Thread, Event
import pandas as pd
import datetime
import time
import os
import socket

sys.path.append('..')
from EITPose.DataVisualizer import MainWindow

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
                


    def send_data(self, head_pos, head_rot, hand_pos, hand_rot, hand_pos_pred, hand_rot_pred, joint_data_ground, joint_data_pred):
        pos = f"{head_pos[0]}, {head_pos[1]}, {head_pos[2]}"
        rot = f"{head_rot[0]}, {head_rot[1]}, {head_rot[2]}, {head_rot[3]}"
        head_strings = '#'.join([pos,rot])
        joint_strings = [f"{q[0]}, {q[1]},{q[2]},{q[3]}" for q in joint_data_ground]
        joint_strings.insert(0, f"{hand_pos[0]}, {hand_pos[1]}, {hand_pos[2]}")
        joint_strings.insert(1, f"{hand_rot[0]}, {hand_rot[1]}, {hand_rot[2]}, {hand_rot[3]}")
        ground = '#'.join(joint_strings)
        joint_strings = [f"{q[0]}, {q[1]},{q[2]},{q[3]}" for q in joint_data_pred]
        joint_strings.insert(0, f"{hand_pos_pred[0]}, {hand_pos_pred[1]}, {hand_pos_pred[2]}")
        joint_strings.insert(1, f"{hand_rot_pred[0]}, {hand_rot_pred[1]}, {hand_rot_pred[2]}, {hand_rot_pred[3]}")
        pred = '#'.join(joint_strings) 
        s = '!'.join([head_strings, ground, pred]) + '$'
        #print("sending from unity1")
        self.conn.send(s.encode('utf8'))
        #print("sending from unity2")

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




# Run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    unityp = True

    if unityp:
        unityconnected = Event()
        endevent = Event()
        unity = UnityManager()
        unity_thread = Thread(target=unity.run, daemon=True, args = (unityconnected,))
        unity_thread.start()
        unityconnected.wait()
        ex = MainWindow(unity)
    else:
        ex = MainWindow()
    
    # Install the event filter for the application
    
    # app.set_main_window(ex)

    # app.installEventFilter(ex)


    style = 'dark'
    if style == 'dark':
        app.setStyleSheet("""
        QWidget { 
            background-color: white; 
        }
        QPushButton {
            color: white;
            background-color: gray;
            border: 1px solid white;
            border-radius: 10px;   /* rounded corners */
            padding: 30px;
            font: 20px "Helvetica";
        }
        QPushButton:hover {
            background-color: darkgray;
        }
        QPushButton:checked {
            background-color: black;
        }
        QPushButton:pressed {
            background-color: black;
        }

        QRadioButton {
            color: white;
            background-color: gray;
            border: 1px solid white;
            border-radius: 10px;   /* rounded corners */
            padding: 30px;
            font: 20px "Helvetica";
        }
        QRadioButton:hover {
            background-color: darkgray;
        }
        QRadioButton:checked {
            background-color: black;
        }
        QRadioButton:pressed {
            background-color: black;
        }

        QLabel {
            color: black;
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

        QRadioButton {
            color: black;
            background-color: lightgray;
            border: 1px solid black;
            border-radius: 10px;   /* rounded corners */
            padding: 5px;
            font: 14px "Helvetica";
        }
        QRadioButton:hover {
            background-color: darkgray;
        }
        QRadioButton:checked {
            background-color: red;
        }
        QRadioButton:pressed {
            background-color: darkred;
        }

        QLabel {
            color: black;
        }
        """)


    sys.exit(app.exec_())
