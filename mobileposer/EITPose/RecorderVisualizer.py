from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QGridLayout, QPushButton, QLineEdit
from PyQt5.QtGui import QFont, QFontDatabase, QColor, QPainter, QImage, QPixmap
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np
import cv2
import datetime
import random
import time
import asyncio
import os


class SimplePoints3DViewer(gl.GLViewWidget):
    """
    A minimalist 3D point visualization widget that displays three points in 3D space.
    
    This class visualizes three 3D points, each with a different color.
    It can be updated with new position data and integrated into existing PyQt applications.
    """
    
    def __init__(self, parent=None, point_size=10, grid_size=20):
        """
        Initialize the 3D points viewer.
        
        Args:
            parent: Parent widget
            point_size: Size of the points
            grid_size: Size of the grid
        """
        super().__init__(parent)
        
        self.point_size = point_size
        
        # Initialize the viewer
        self._setup_viewer(grid_size)
        
        # Create scatter items for current positions
        self.scatter_items = [
            gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=color, size=self.point_size)
            for color in [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]  # Red, Green, Blue
        ]
        
        # Add scatter items to the view
        for scatter in self.scatter_items:
            self.addItem(scatter)
    
    def _setup_viewer(self, grid_size):
        """Set up the 3D viewer with grid and axes."""
        # Add coordinate grid
        grid = gl.GLGridItem()
        grid.setSize(grid_size, grid_size, grid_size)
        grid.setSpacing(1, 1, 1)
        self.addItem(grid)
        
        # Add coordinate axes
        axis_x = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [10, 0, 0]]), color=(1, 0, 0, 1), width=2)
        axis_y = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 10, 0]]), color=(0, 1, 0, 1), width=2)
        axis_z = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, 10]]), color=(0, 0, 1, 1), width=2)
        self.addItem(axis_x)
        self.addItem(axis_y)
        self.addItem(axis_z)
    
    def update_positions(self, positions):
        """
        Update the visualization with new positions.
        
        Args:
            positions: List of three numpy arrays, each containing a 3D position.
                      Each array should be shape (3,) or (1, 3) representing x, y, z coordinates.
        """
        if len(positions) != 3:
            raise ValueError(f"Expected 3 position arrays, got {len(positions)}")
        
        for i, pos_array in enumerate(positions):
            # Ensure the position array is the right shape
            if len(pos_array.shape) == 1:
                # If it's a single 1D array with 3 elements, reshape it to (1, 3)
                if pos_array.shape[0] == 3:
                    pos_array = pos_array.reshape(1, 3)
                else:
                    raise ValueError(f"Position array {i} has invalid shape: {pos_array.shape}")
            else:
                # If it's already 2D, ensure it's (1, 3)
                if pos_array.shape != (1, 3):
                    raise ValueError(f"Position array {i} must have shape (1, 3), got {pos_array.shape}")
            
            # Update current position
            self.scatter_items[i].setData(pos=pos_array)
    
    def set_point_size(self, size):
        """Set the size of the points."""
        self.point_size = size
        for scatter in self.scatter_items:
            scatter.setData(size=size)
    
    def set_point_colors(self, colors):
        """
        Set custom colors for the three points.
        
        Args:
            colors: List of three colors, each as an RGBA tuple (r, g, b, a) with values 0-1
        """
        if len(colors) != 3:
            raise ValueError(f"Expected 3 colors, got {len(colors)}")
            
        for i, color in enumerate(colors):
            self.scatter_items[i].setData(color=color)



class GLWidget(gl.GLViewWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.overlay_text = {}

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Helvetica", 16))

        for text, x, y, size in self.overlay_text.values():
            painter.setFont(QFont("Helvetica", size))
            painter.drawText(x, y, text)

    def update_overlay_text(self, key, text, x, y, size):
        self.overlay_text.update({key: (text, x, y, size)})
        self.update()



class InfoWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Create a QVBoxLayout
        self.layout = QVBoxLayout()

        # Create a QLabel
        self.recording_status = QLabel()
        self.bt_data_length = QLabel()
        self.eit_data_length = QLabel()
        self.imu_data_length = QLabel()
        self.mp_data_length = QLabel()
        self.bt_connected_status = QLabel()
        self.bt_fps = QLabel()
        self.mp_fps = QLabel()
        self.gui_fps = QLabel()

        # Add the QLabel to the QVBoxLayout
        self.layout.addWidget(self.recording_status)
        self.layout.addWidget(self.bt_data_length)
        self.layout.addWidget(self.eit_data_length)
        self.layout.addWidget(self.imu_data_length)
        self.layout.addWidget(self.mp_data_length)
        self.layout.addWidget(self.bt_connected_status)
        self.layout.addWidget(self.bt_fps)
        self.layout.addWidget(self.mp_fps)
        self.layout.addWidget(self.gui_fps)

        # Set the layout of the QWidget
        self.setLayout(self.layout)

    def set_recording_status(self, text):
        # Update the text of the QLabel
        self.recording_status.setText(text)

    def set_bt_data_length(self, text):
        # Update the text of the QLabel
        self.bt_data_length.setText(text)

    def set_eit_data_length(self, text):
        # Update the text of the QLabel
        self.eit_data_length.setText(text)

    def set_imu_data_length(self, text):
        # Update the text of the QLabel
        self.imu_data_length.setText(text)
    
    def set_mp_data_length(self, text):
        # Update the text of the QLabel
        self.mp_data_length.setText(text)

    def set_bt_connected_status(self, text):
        # Update the text of the QLabel
        self.bt_connected_status.setText(text)

    def set_bt_fps(self, text):
        # Update the text of the QLabel
        self.bt_fps.setText(text)

    def set_mp_fps(self, text):
        # Update the text of the QLabel
        self.mp_fps.setText(text)

    def set_gui_fps(self, text):
        # Update the text of the QLabel
        self.gui_fps.setText(text)

class ArmWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.view = gl.GLViewWidget()
        self.layout.addWidget(self.view)

        # Set top-down camera position and view
        self.camera_distance = 5
        # self.view.setCameraPosition(distance=1, elevation=90, azimuth=0)  # Top-down view

        # Create scatter plot for joints
        self.joints_plot = gl.GLScatterPlotItem(size=.1, color=(1, 0, 0, 1), pxMode=False)
        self.view.addItem(self.joints_plot)

        # Create line plot for bones
        self.bones_plot = gl.GLLinePlotItem(color=(1, 1, 1, 1), width=1)
        self.view.addItem(self.bones_plot)


        # Placeholder for rarm_joints data
        self.rarm_joints = np.zeros((5, 3))  # Initialize to zeros, replace with actual data

    def update_plot(self, arm_data):
        self.joints_plot.setData(pos=arm_data)  # Update joints
        self.bones_plot.setData(pos=arm_data)   # Update bones




class MainWindow(QMainWindow):
    def __init__(self, data_dict, save_data_dict, clear_data_dict, teensybt, mobileimu, unity):
        super().__init__()

        self.data_dict = data_dict
        self.save_data_dict = save_data_dict
        self.clear_data_dict = clear_data_dict
        #self.mphands = mphands # used for getting cv2 image from mediapipe processing
        self.teensybt = teensybt # used for getting quaternion data
        self.mobileimu = mobileimu
        self.unity = unity

        self.recording_status = False

        # Font sizes
        self.title_font_size = 16
        self.stat_font_size = 10

        # GUI FPS
        self.timestamp_prev = 0
        self.fps = 0
        self.fps_total = np.zeros(100)
        self.i = 0


        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create a vertical layout and set it as the layout of the central widget
        self.vertical_layout = QVBoxLayout()
        self.central_widget.setLayout(self.vertical_layout)

        self.viewer = SimplePoints3DViewer()
        self.viewer.setMinimumSize(400,300)

        # Create the first row
        self.first_row_layout = QHBoxLayout()
        # self.start_button = QPushButton('Start')
        # self.stop_button = QPushButton('Stop')
        self.record_button = QPushButton('Start Recording')
        self.record_button.setCheckable(True)
        self.clear_button = QPushButton('Clear')
        self.textbox = QLineEdit()
        self.save_button = QPushButton('Save')
        self.gesture_set_button = QPushButton('Gesture Set')
        self.gesture_set_button.setCheckable(True)
        # self.first_row_layout.addWidget(self.start_button)
        # self.first_row_layout.addWidget(self.stop_button)
        self.first_row_layout.addWidget(self.record_button)
        self.first_row_layout.addWidget(self.clear_button)
        self.first_row_layout.addWidget(self.textbox)
        self.first_row_layout.addWidget(self.save_button)
        self.first_row_layout.addWidget(self.gesture_set_button)
        self.vertical_layout.addLayout(self.first_row_layout)

        # self.start_button.clicked.connect(self.start_button_clicked)
        # self.stop_button.clicked.connect(self.stop_button_clicked)
        self.record_button.clicked.connect(self.record_button_clicked)
        self.save_button.clicked.connect(self.save_button_clicked)
        self.clear_button.clicked.connect(self.clear_button_clicked)
        self.save_button.clicked.connect(self.save_button_clicked)

        # Create the second row
        self.second_row_layout = QHBoxLayout()
        
        self.second_row_widget_1 = InfoWidget() # leave this open for random stats
        self.second_row_widget_1.sizeHint = lambda: pg.QtCore.QSize(200, 600)  # Set a larger window size
        self.second_row_widget_1.set_recording_status("Recording Status: False")
        self.second_row_widget_1.set_bt_data_length("Teensy Data Length: 0")
        self.second_row_widget_1.set_eit_data_length("EIT Data Length: 0")
        self.second_row_widget_1.set_imu_data_length("IMU Data Length: 0")
        self.second_row_widget_1.set_mp_data_length("MP Hands Data Length: 0")
        self.second_row_widget_1.set_bt_connected_status("Bluetooth Status: False")
        self.second_row_widget_1.set_bt_fps("Bluetooth FPS: 0")
        self.second_row_widget_1.set_mp_fps("Mediapipe FPS: 0")
        self.second_row_widget_1.set_gui_fps("GUI FPS: 0")

        self.second_row_widget_2 = pg.PlotWidget()
        self.second_row_widget_2.showGrid(x = True, y = True)
        self.second_row_widget_2.setLabel('left', 'EIT Signal', units ='y')
        self.second_row_widget_2.setLabel('bottom', 'Linux Timestamp', units ='s')
        # self.graphWidget1.setXRange(0, 10)
        # self.graphWidget1.setYRange(0, 20)
        self.second_row_widget_2.setTitle("EIT Signals")
        # self.eit_plots = [self.second_row_widget_2.plot() for _ in range(64)]
        pen_colors = ['r', 'g', 'b', 'y', 'm', 'c', (255, 165, 0), (128, 0, 128)]
        self.eit_plots = [self.second_row_widget_2.plot(pen=pen_colors[i]) for i in range(8)]
        # self.eit_plots = [self.second_row_widget_2.plot() for _ in range(8)]

        self.second_row_layout.addWidget(self.second_row_widget_1)
        self.second_row_layout.addWidget(self.second_row_widget_2)
        # self.second_row_layout.addWidget(self.second_row_widget_3)
        self.vertical_layout.addLayout(self.second_row_layout)


        self.third_row_layout = QHBoxLayout()
        self.imu_layout = QVBoxLayout()
        self.acc_label = QLabel()
        self.quat_label = QLabel()
        self.body_label = QLabel()
        self.imu_layout.addWidget(self.acc_label)
        self.imu_layout.addWidget(self.quat_label)
        self.imu_layout.addWidget(self.body_label)
        self.acc_label.setText(f"Acc: none yet")
        self.quat_label.setText(f"Quat: none yet")
        self.body_label.setText(f"Body Length: none yet")
        self.third_row_widget_2 = GLWidget()
        self.third_row_widget_2.update_overlay_text("title", "Quest Handpose (Ground Truth)", 10, 30, self.title_font_size)
        self.third_row_widget_2.sizeHint = lambda: pg.QtCore.QSize(800, 600)  # Set a larger window size
        self.arm_widget = ArmWidget()
        self.arm_widget.sizeHint = lambda: pg.QtCore.QSize(800, 600) 

        
        self.third_row_layout.addLayout(self.imu_layout)
        #self.third_row_layout.addWidget(self.viewer)
        self.third_row_layout.addWidget(self.arm_widget)

        self.vertical_layout.addLayout(self.third_row_layout)

        # Create a QTimer for animation updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.animation_loop)
        self.timer.start(30)  # 20 milliseconds

    async def disconnect_bt(self):
        # if self.teensybt.get_connected_status():
        await self.teensybt.stop()

    def closeEvent(self, event):
        # loop = asyncio.get_event_loop()
        # loop.run_until_complete(self.disconnect_bt())
        event.accept()  # Let the window close




    def record_button_clicked(self, pressed):
        if pressed:
            self.recording_status = True
            self.timer_gesture = time.time()
            self.record_button.setText('Stop Recording')
            self.teensybt.collect_data = True
            self.mobileimu.collect_data = True
            self.unity.collect_data = True
        else:
            self.recording_status = False
            self.record_button.setText('Start Recording')
            self.teensybt.collect_data = False
            self.mobileimu.collect_data = False
            self.unity.collect_data = False

    def record_data(self):
        linux_time = time.time()
        self.teensybt.record_data(linux_time)
        self.mobileimu.record_data(linux_time)
        self.unity.record_data(linux_time)


    def save_button_clicked(self):
        # This function will be called when the stop button is clicked
        text = self.textbox.text()
        self.save_data_dict(text)


    def clear_button_clicked(self):
        # This function will be called when the stop button is clicked
        self.stop_data_collection()
        self.recording_status = False
        self.clear_data_dict()
        self.total_timer_gesture = 0
        self.randomize_gesture_set()

    def resizeEvent(self, event):
        super(MainWindow, self).resizeEvent(event)
        size = event.size()
        self.update_plot_size(size)
    
    def update_plot_size(self, size):
        plot_width = int(size.width() / 2)  # Divide by 2 to fit two plots side by side
        plot_height = int(size.height())

        self.third_row_widget_2.setGeometry(0, 0, plot_width, plot_height)
        # self.plot2.setGeometry(plot_width, 0, plot_width, plot_height)


    # Animation loop triggered by QTimer
    def animation_loop(self):
        # Get new hand landmarks for plot1, add this also when we get handtracking
        #x1, y1, z1, fps_mediapipe = self.generate_hand_landmarks(0)

        # Update plot1
        #add this back in when you get hand position from quest
        #self.update_plot(self.third_row_widget_2, x1, y1, z1, fps_mediapipe) # plot1


        recent_imu = self.mobileimu.get_recent_data_imu()
        recent_arm = self.mobileimu.get_recent_data_arm()
        teensybt_recent_eit = self.teensybt.get_recent_data_eit()
        pose,tran = self.mobileimu.get_recent_predictions()
        recent_hand = self.unity.get_recent_joint_data()
        body_pos, body_rot = self.unity.get_recent_body_data()
        head_pos, head_rot = self.unity.get_recent_head_data()
        hand_pos, hand_rot = self.unity.get_recent_hand_data()
        ball_pos, _, _ = self.unity.get_recent_gesture_data()
        #self.unity.send_data(pose, tran, recent_hand)
        try:
            eit = [dat[1] for dat in teensybt_recent_eit[:100]]
            timestamps = [dat[0] for dat in teensybt_recent_eit[:100]]
            # [self.eit_plots[i].setData(timestamps, [eit_val[i] for eit_val in eit]) for i in range(64)]
            [self.eit_plots[i].setData(timestamps, [sum(eit_val[0+i:64:8])/8*100 for eit_val in eit]) for i in range(8)]
            self.acc_label.setText(f"Acc: {recent_imu[0]}")
            self.quat_label.setText(f"Quat: {recent_imu[1]}")
            body_length = len(body_pos)
            self.body_label.setText(f"Body Length: {body_length}")


            self.update()
        except Exception as e:
            # print(e)
            pass
        
        timestamp = datetime.datetime.now()
        # Get new hand landmarks for plot1, add this also when we get handtracking
        #x1, y1, z1, fps_mediapipe = self.generate_hand_landmarks(0)

        # Update plot1
        #add this back in when you get hand position from quest
        #self.update_plot(self.third_row_widget_2, x1, y1, z1, fps_mediapipe) # plot1
        #self.viewer.update_positions([head_pos, hand_pos, ball_pos])

        #plot new arm data
        self.arm_widget.update_plot(recent_arm)



        if self.timestamp_prev == 0:
                self.timestamp_prev = timestamp
                return
        else:
            # if self.fps == 0:
            fps = 1/((timestamp - self.timestamp_prev).total_seconds())
            self.timestamp_prev = timestamp
            self.i += 1
            self.fps_total[int(self.i%100)] = fps
            self.fps = sum(self.fps_total[self.fps_total != 0]) / sum(self.fps_total != 0)
            # print(self.fps)


        self.second_row_widget_1.set_recording_status("Recording Status: " + str(self.recording_status))
        if self.teensybt.DATA_DICT_ENTRY in self.data_dict:
            self.second_row_widget_1.set_bt_data_length("Teensy Data Length: " + str(len(self.data_dict[self.teensybt.DATA_DICT_ENTRY])))
        else:
            self.second_row_widget_1.set_bt_data_length("Teensy Data Length: " + str(0))
        if self.teensybt.DATA_DICT_ENTRY_EIT in self.data_dict:
            self.second_row_widget_1.set_eit_data_length("EIT Data Length: " + str(len(self.data_dict[self.teensybt.DATA_DICT_ENTRY_EIT])))
        else:
            self.second_row_widget_1.set_eit_data_length("EIT Data Length: " + str(0))
        if self.mobileimu.IMU_DATA_DICT_ENTRY in self.data_dict:
            self.second_row_widget_1.set_imu_data_length("IMU Data Length: " + str(len(self.data_dict[self.mobileimu.IMU_DATA_DICT_ENTRY])))
        else:
            self.second_row_widget_1.set_imu_data_length("IMU Data Length: " + str(0))
        """ update this too
        if self.mphands.DATA_DICT_ENTRY in self.data_dict:
            self.second_row_widget_1.set_mp_data_length("MP Hands Data Length: " + str(len(self.data_dict[self.mphands.DATA_DICT_ENTRY])))
        else:
            self.second_row_widget_1.set_mp_data_length("MP Hands Data Length: " + str(0))
        """
            
        self.second_row_widget_1.set_bt_connected_status("Bluetooth Status: " + str(self.teensybt.get_connected_status()))
        self.second_row_widget_1.set_bt_fps("Teensy FPS: " + str(int(self.teensybt.get_fps())) + ", " + str(int(self.teensybt.get_fps_imu())) + ", " + str(int(self.teensybt.get_fps_eit())))
        #self.second_row_widget_1.set_mp_fps("Mediapipe FPS: " + str(int(self.mphands.get_fps())))
        self.second_row_widget_1.set_gui_fps("GUI FPS: " + str(int(self.fps)))
        