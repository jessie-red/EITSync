import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QFileDialog, QSlider, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QRadioButton, QButtonGroup, QGraphicsView, QGraphicsScene)
from PyQt5.QtGui import QFont, QFontDatabase, QColor, QPainter, QImage, QPixmap, QPolygonF, QLinearGradient, QPen, QBrush
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import GLViewWidget
from PyQt5.QtCore import Qt, QTimer, QEvent, QPointF
import pyqtgraph as pg

import numpy as np
import pandas as pd

import scipy.io

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from EITPose.EITVisualizer import EITVisualization








class MainWindow(QMainWindow):
    def __init__(self, unity=False):
        super().__init__()

        self.gesture_dictionary = {
            0: "fist",
            1: "spiderman",
            2: "OK",
            3: "claw",
            4: "stretch",
            5: "point",
            6: "pinch",
            7: "close",
            8: "three point",
            9: "gun",
            10: "six",
            11: "thumbs up",
            12: "relax",
            13: "open",
            14: "down",
            15: "up",
            16: "left",
            17: "right",
            18: None,
        }

        self.picture_to_gesture = {
            'gesture2': "claw",
            'gesture3': "spiderman",
            'gesture4': "six",
            'gesture6': "OK",
            'gesture7': "point",
            'gesture8': "gun",
            'gesture9': "thumbs up",
            'gesture10': "stretch",
            'gesture11': "fist",
            'gesture12': "three point",
            'gesture13': "pinch",
            'gesture14': "close",
            'wrist1': "right",
            'wrist2': "up",
            'wrist3': "down",
            'wrist4': "open",
            'wrist5': "left",
        }

        self.unity = unity
        self.pred = True


       
        self.eit_plot = pg.PlotWidget()
        self.eit_plot.showGrid(x = True, y = True)
        self.eit_plot.setLabel('left', 'EIT Signal', units ='y')
        self.eit_plot.setLabel('bottom', 'Linux Timestamp', units ='s')
        pen_colors = ['r', 'g', 'b', 'y', 'm', 'c', (255, 165, 0), (128, 0, 128)]
        self.eit_plots = [self.eit_plot.plot(pen=pen_colors[i]) for i in range(8)]

        self.acc_plot = pg.PlotWidget()
        self.acc_plot.showGrid(x = True, y = True)
        self.acc_plot.setLabel('left', 'Value', units ='y')
        self.acc_plot.setLabel('bottom', 'Linux Timestamp', units ='s')
        self.acc_plots = [self.acc_plot.plot(pen=pen_colors[i]) for i in range(6)]


        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setTickPosition(QSlider.NoTicks)  # No tick marks
        self.timeline.setSingleStep(1)  # Set the single step to 1 for fine control
        self.timeline.setMinimum(0)  # Start of your range
        self.timeline.setMaximum(4999)  # End of your range (5000 points)
        self.timeline.valueChanged.connect(self.timeline_value_changed)

        self.eit_view = EITVisualization()
        self.eit_view.sizeHint = lambda: pg.QtCore.QSize(800, 400)

        self.label_layout = QVBoxLayout()
        self.gesture_label = QLabel(f"Gesture: ")
        self.orientation_label = QLabel(f"Orientation: ")
        self.ball_label = QLabel(f"Ball: ")
        self.label_layout.addWidget(self.gesture_label)
        self.label_layout.addWidget(self.orientation_label)
        self.label_layout.addWidget(self.ball_label)


        control_buttons = QHBoxLayout()
        self.load_button = QPushButton('Load File')
        self.load_button.clicked.connect(self.open_file_dialog)
        self.play_button = QPushButton('Play', checkable=True)
        self.play_button.clicked.connect(self.play_button_clicked)

        control_buttons.addWidget(self.load_button)
        control_buttons.addWidget(self.play_button)

        self.annotation_enabled = False
        self.playing = False



        # Layout setup
        layout = QVBoxLayout()
        plotsLayout = QHBoxLayout()
        #plotsLayout.addWidget(self.viewer)
        plotsLayout.addLayout(self.label_layout)
        plotsLayout.addWidget(self.eit_view)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        layout.addLayout(plotsLayout)
        
        layout.addWidget(self.acc_plot)
        #layout.addLayout(graph_layout)
        layout.addWidget(self.eit_plot)
        layout.addWidget(self.timeline)
        #layout.addWidget(self.view)
        layout.addLayout(control_buttons)
        
        

        self.labeled_data = None
        self.original_hand_data = None
        self.prelabeled_gestures = None
        self.index_value = 0

        # Set the central widget and show the GUI
        self.setCentralWidget(central_widget)
        self.show()

        # Create a QTimer for animation updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.animation_loop)
        # self.timer.start(30)  # 20 milliseconds



        

    def open_file_dialog(self):
        # This method will be called when the 'Load file' button is clicked
        options = QFileDialog.Options()
        # Uncomment the following line if you want a native file dialog.
        # options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "All Files (*);;PKL Files (*.pkl)", options=options)
        if file_name:
            # Add code here to handle the file loading
            self.load_handpose_data(file_name)
            self.load_button.setText("Loaded: " + file_name.split('/')[-1])

    def load_handpose_data(self, file_name):
        # Add code here to load the handpose data
        data = pd.read_pickle(file_name)
        if isinstance(data, dict):
            if 'ground_truth_rot' in data:
                #print(data['ground_truth_rot'].shape)
                self.joint_data = data['ground_truth_rot'].reshape(-1, 19,4)
            else:
                raise ValueError("Expected 'ground_truth_rot' in the dictionary")
            if 'pred_rot' in data:
                #print(data['ground_truth_rot'].shape)
                self.joint_data_pred = data['pred_rot'].reshape(-1, 19,4)
            else:
                raise ValueError("Expected 'pred_rot' in the dictionary")
            self.eit_data = None
            self.timestamps = np.array([i for i in range(len(self.joint_data))])
            self.timeline.setMaximum(len(self.timestamps)-1)
            return

        data = data[10:]
        #print(data.dtypes)
        

        if 'acc' in data.columns:
            self.acc = np.vstack([tensor.numpy() for tensor in data['acc']])
        else:
            self.acc = None
        if 'ori' in data.columns:
            self.ori = data['ori'].to_numpy()
        else:
            self.ori = None
        if 'head_pos' in data.columns:
            self.head_pos = np.stack(data['head_pos'].to_numpy())
        else:
            self.head_pos = None
        if 'head_rot' in data.columns:
            self.head_rot = np.stack(data['head_rot'].to_numpy())
        else:
            self.head_rot = None
        if 'hand_pos' in data.columns:
            self.hand_pos = np.stack(data['hand_pos'].to_numpy())
        else:
            self.hand_pos = None
        if 'hand_rot' in data.columns:
            self.hand_rot = np.stack(data['hand_rot'].to_numpy())
        else:
            self.hand_rot = None
        if 'hand_pos_pred' in data.columns:
            self.hand_pos_pred = np.stack(data['hand_pos_pred'].to_numpy())
        else:
            self.hand_pos_pred = None
        if 'hand_rot_pred' in data.columns:
            self.hand_rot_pred = np.stack(data['hand_rot_pred'].to_numpy())
        else:
            self.hand_rot_pred = None
        if 'joint_data' in data.columns:
            self.joint_data = data['joint_data'].to_numpy()
        else:
            self.joint_data = None
        if 'joint_data_pred' in data.columns:
            self.joint_data_pred = np.stack(data['joint_data_pred'].tolist())
        else:
            self.joint_data_pred = None
        if 'ball_pos' in data.columns:
            self.ball_pos = np.stack(data['ball_pos'].to_numpy())
        else:
            self.ball_pos = None
        if 'gesture' in data.columns:
            self.gesture_actual = data['gesture'].to_numpy()
        else:
            self.gesture_actual = None
        if 'orientation' in data.columns:
            self.orientation = data['orientation'].to_numpy()
        else:
            self.orientation = None
        if 'eit_data' in data.columns:
            self.eit_data = np.stack(data['eit_data'].tolist())
        elif 'data' in data.columns:
            self.eit_data = np.stack(data['data'].tolist())

        if self.joint_data_pred is None:
            self.pred = False

        #print(self.eit_data.shape)
        print(self.joint_data_pred.shape)
        print(self.joint_data_pred[0].shape)


        self.eit_data = self.eit_data - np.mean(self.eit_data, axis=0)
        self.eit_view.calc_perms(self.eit_data)
        self.timestamps = np.array([i for i in range(len(self.eit_data))])
        self.timeline.setMaximum(len(self.timestamps)-1)


        for i in range(8):
            self.eit_plots[i].setData(self.timestamps[0:100], self.eit_data[0:100, 40+i])
        self.eit_view.plot_tripcolor(0)


    def play_button_clicked(self, pressed):
        # This method will be called when the 'Play' button is clicked
        # You will need to implement the logic to play the handpose data
        if pressed:
            self.timer.start(50)
            self.playing = True
            if self.annotation_enabled and self.gesture_group.checkedId() != 12:
                self.start_timestamp = self.timestamps[self.index_value]
        else:
            self.timer.stop()
            self.playing = False
        self.play_button.setText("Play: " + str(pressed))

    def annotate_button_clicked(self, pressed):
        # This method will be called when the 'Annotate' button is clicked
        # You will need to implement the logic to annotate the handpose data
        self.annotation_enabled = pressed
        self.annotate_button.setText("Annotate: " + str(pressed))


    def timeline_value_changed(self, value):
        self.index_value = value

        #self.viewer.update_positions([self.head_pos[value,:], self.hand_pos[value,:], self.ball_pos[value,:]])

        if self.eit_data is None:
            dummy_rot = np.array([0,0,0,0])
            self.unity.send_data(np.array([0,1.16,0]), dummy_rot, np.array([0,.8,0]), dummy_rot,
                np.array([-.2,.8,0]), dummy_rot, self.joint_data[value], self.joint_data_pred[value])
            return


        for i in range(8):
            self.eit_plots[i].setData(self.timestamps[value:min(100+value, len(self.timestamps)-1)], self.eit_data[value:min(100+value, len(self.timestamps)-1), 40+i])
        for i in range(self.acc.shape[1]):
            self.acc_plots[i].setData(self.timestamps[value:min(100+value, len(self.timestamps)-1)], self.acc[value:min(100+value, len(self.timestamps)-1), i])
        self.eit_view.plot_tripcolor(value)
        self.gesture_label.setText(f"Gesture: {self.gesture_actual[value]}")
        self.orientation_label.setText(f"Orientation: {self.orientation[value]}")
        self.ball_label.setText(f"Ball: {self.ball_pos[value]}")
        if self.unity:
            if self.pred:
                self.unity.send_data(self.head_pos[value], self.head_rot[value], self.hand_pos[value], self.hand_rot[value],
                self.hand_pos_pred[value], self.hand_rot_pred[value], self.joint_data[value], self.joint_data_pred[value])
            else:
                self.unity.send_data(self.head_pos[value], self.head_rot[value], self.hand_pos[value], self.hand_rot[value],
                self.hand_pos[value], self.hand_rot[value], self.joint_data[value], self.joint_data[value])



    def animation_loop(self):
        # This method will be called by the QTimer to update the animation
        # You will need to implement the logic to update the animation
        if self.playing:

            self.index_value += 1
            if self.index_value >= len(self.timestamps):
                self.play_button.setChecked(False)
                self.playing = False
                self.timer.stop()
                self.index_value -= 1
                self.timeline.setValue(self.index_value)
                self.timeline_value_changed(self.index_value)
                return
            self.timeline.setValue(self.index_value)
            self.timeline_value_changed(self.index_value)
            if self.annotation_enabled:
                if self.index_value < len(self.labeled_data):
                    self.labeled_data[self.index_value] = None if self.gesture_group.checkedId() == 18 else self.gesture_group.checkedId()
                else:
                    self.play_button.setChecked(False)
                    self.playing = False
                    self.timer.stop()
            if self.annotation_enabled and self.gesture_group.checkedId() == 18:
                self.start_timestamp = self.timestamps[self.index_value]
            if self.annotation_enabled and self.gesture_group.checkedId() != 18 and self.timestamps[self.index_value] - self.start_timestamp > 1.5:
                self.none_button.setChecked(True)
                self.play_button_clicked(False)

    def keyPressEvent(self, event):
        # Override keyPressEvent to respond to the space bar
        if event.key() == Qt.Key_F:
            # Toggle the play button's checked state
            # self.play_button.setChecked(not self.play_button.isChecked())
            self.play_button_clicked(not self.playing)
        else:
            super().keyPressEvent(event)  # Call the parent class method to ensure default behavior


