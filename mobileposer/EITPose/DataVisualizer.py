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



class MainWindow(QMainWindow):
    def __init__(self):
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

        control_buttons = QHBoxLayout()
        self.load_button = QPushButton('Load File')
        self.load_button.clicked.connect(self.open_file_dialog)
        self.play_button = QPushButton('Play', checkable=True)
        self.play_button.clicked.connect(self.play_button_clicked)

        control_buttons.addWidget(self.load_button)
        control_buttons.addWidget(self.play_button)

        self.annotation_enabled = False
        self.playing = False

        self.viewer = SimplePoints3DViewer()
        self.viewer.setMinimumSize(400,300)



        # Layout setup
        layout = QVBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(layout)
        layout.addWidget(self.viewer)
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

        if 'acc' in data.columns:
            self.acc = np.stack(data['acc'].to_numpy())
        else:
            self.acc = None
        if 'ori' in data.columns:
            self.ori = np.stack(data['ori'].to_numpy())
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
            self.hand_rpt = None
        if 'ball_pos' in data.columns:
            self.ball_pos = np.stack(data['ball_pos'].to_numpy())
        else:
            self.ball_pos = None
        if 'gesture' in data.columns:
            self.gesture_actual = data['gesture'].to_numpy()
        else:
            self.gesture_actual = None
        if 'eit_data' in data.columns:
            self.eit_data = np.stack(data['eit_data'].tolist())
        elif 'data' in data.columns:
            self.eit_data = np.stack(data['data'].tolist())

        #print(self.eit_data.shape)
        #print(self.gesture_actual.shape)
        #print(self.head_pos.shape)
        #print(self.hand_pos.shape)
        #print(self.acc[1].shape)
        #print(self.ori[1].shape)

        self.eit_data = self.eit_data - np.mean(self.eit_data, axis=0)
        self.timestamps = np.array([i for i in range(len(self.eit_data))])
        self.timeline.setMaximum(len(self.timestamps)-1)

        for i in range(8):
            self.eit_plots[i].setData(self.timestamps[0:100], self.eit_data[0:100, 40+i])


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

        self.viewer.update_positions([self.head_pos[value,:], self.hand_pos[value,:], self.ball_pos[value,:]])

        for i in range(8):
            self.eit_plots[i].setData(self.timestamps[value:min(100+value, len(self.timestamps)-1)], self.eit_data[value:min(100+value, len(self.timestamps)-1), 40+i])
        for i in range(self.acc.shape[1]):
            self.acc_plots[i].setData(self.timestamps[value:min(100+value, len(self.timestamps)-1)], self.acc[value:min(100+value, len(self.timestamps)-1), i])



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
    def is_point_inside_polygon(self, point, polygon):
        """
        Determine if a point lies inside a polygon using the ray casting algorithm.
        
        Parameters:
        point: tuple of (x, y) coordinates for the test point
        polygon: list of tuples, each containing (x, y) coordinates for polygon vertices
        
        Returns:
        bool: True if point is inside the polygon, False otherwise
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        # Get the first point of the polygon
        p1x, p1y = polygon[0]
        
        # Iterate through each edge of the polygon
        for i in range(polygon.shape[0]):
            # Get the next point (wrap around to first point if at end)
            p2x, p2y = polygon[i,:]
            
            # Check if point is above the minimum y coordinate of line segment
            if y > min(p1y, p2y):
                # Check if point is below the maximum y coordinate of line segment
                if y <= max(p1y, p2y):
                    # Check if point is to the left of maximum x coordinate of line segment
                    if x <= max(p1x, p2x):
                        # Calculate intersection point
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        
                        # If either the point is to the left of the line segment or
                        # the point is directly on the line segment
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            
            # Move to the next point
            p1x, p1y = p2x, p2y
        
        return inside
    def running_average(self, x, window_size):
        """
        Calculate the running average of a numpy array along the first dimension.
        
        Parameters:
        -----------
        x : array_like
            Input array (1D or 2D)
        window_size : int
            Size of the moving window
            
        Returns:
        --------
        numpy.ndarray
            Array of running averages with same number of dimensions as input
        """
        # Convert input to numpy array if it isn't already
        x = np.asarray(x)
        
        # Handle both 1D and 2D arrays
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        elif x.ndim > 2:
            raise ValueError("Input array must be 1D or 2D")
        
        # Create a padded array to handle edge cases
        pad_width = ((window_size - 1, 0), (0, 0))
        x_padded = np.pad(x, pad_width, mode='edge')
        
        # Create a 3D array of sliding windows
        shape = (x.shape[0], window_size, x.shape[1])
        strides = (x.itemsize * x.shape[1],  # stride along first dim
                x.itemsize * x.shape[1],   # stride for window
                x.itemsize)                # stride along second dim
        windows = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
        
        # Calculate the average for each window
        result = np.mean(windows, axis=1)
        
        # Return 1D array if input was 1D
        if x.shape[1] == 1:
            return result.ravel()
        return result


    def keyPressEvent(self, event):
        # Override keyPressEvent to respond to the space bar
        if event.key() == Qt.Key_F:
            # Toggle the play button's checked state
            # self.play_button.setChecked(not self.play_button.isChecked())
            self.play_button_clicked(not self.playing)
        else:
            super().keyPressEvent(event)  # Call the parent class method to ensure default behavior


