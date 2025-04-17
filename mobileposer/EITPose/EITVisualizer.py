import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QSlider, 
                            QVBoxLayout, QWidget, QLabel, QHBoxLayout, 
                            QGroupBox, QGraphicsView, QGraphicsScene)
from PyQt5.QtGui import QFont, QFontDatabase, QColor, QPainter, QImage, QPixmap, QPolygonF, QLinearGradient, QPen, QBrush
from PyQt5.QtCore import Qt, QRect, QPointF
import numpy as np
import scipy.io
from mobileposer.config import *
import pyeit.mesh as mesh
import pyeit.mesh.shape as shape
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
import pyeit.eit.protocol as protocol
from pyeit.eit.fem import EITForward
import pyeit.eit.bp as bp
import pyeit.eit.jac as jac
from pyeit.eit.interp2d import sim2pts
class EITVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.eit_view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.eit_view.setScene(self.scene)
        layout = QVBoxLayout(self)
        layout.addWidget(self.eit_view)
        self.setLayout(layout)
        #initialize pyeit stuff
        self.nelec = 8
        def ellip_fd(pts):
            if pts.ndim == 1:
                pts = pts[np.newaxis]
            a, b = 1.1, .9
            return np.sum((pts / [a, b]) ** 2, axis=1) - 1.0
        
    
        mat = scipy.io.loadmat(paths.mesh_data)
        Jacobian = mat['Jsub']
        armimg = mat['armimg'][0][0]
        fwd_model = armimg[2][0][0]
        nodes = fwd_model[2]
        nodes= np.hstack((nodes,np.zeros((nodes.shape[0],1))))
        elems = fwd_model[3] -1
        elecs = fwd_model[7][0]
        mat_idx = fwd_model[6][0]
        self.mesh_obj = mesh.create(self.nelec)
        self.mesh_obj.node = nodes
        self.mesh_obj.element = elems
        for elecidx, elec in enumerate(elecs):
            elecarray = elec[0][0]
            node = elecarray[round(len(elecarray)/2)-1] - 1
            self.mesh_obj.el_pos[elecidx] = node

        #self.mesh_obj = mesh.create(self.nelec, h0=0.1, fd=ellip_fd, fh = shape.area_uniform)
        self.pts = 100*self.mesh_obj.node
        self.tri = self.mesh_obj.element
        self.protocol_obj = protocol.create(self.nelec, dist_exc=1, step_meas=1, parser_meas="std")
        self.fwd = EITForward(self.mesh_obj, self.protocol_obj)
        self.v0 = self.fwd.solve_eit()
        self.eit = bp.BP(self.mesh_obj, self.protocol_obj)
        self.eit.setup(weight="none")
        self.eit = jac.JAC(self.mesh_obj, self.protocol_obj)
        self.lamb = .007
        self.p = .2
        self.method = "kotre"
        self.eit.setup(p=self.p, lamb=self.lamb, method=self.method, perm=1, jac_normalized=True)
        self.eit.H = self.eit._compute_h(Jacobian, self.p,self.lamb, self.method)
        self.inFDS = mat_idx[1] -1
        self.inFDP = mat_idx[2] -1
        self.inFPL = mat_idx[3] -1

    def create_colormap(self, value):
        """Create a color based on value using a blue-red colormap"""
        
        normalized = (value - self.min_val) / (self.max_val - self.min_val)
        # Create a smooth transition from blue to red
        if normalized < 0.5:
            # Blue to white
            blue = 255
            red = green = int(510 * normalized)
        else:
            # White to red
            red = 255
            blue = green = int(510 * (1 - normalized))
        return QColor(red, green, blue)

    def plot_tripcolor(self, permidx):
        """Create a tripcolor plot using native Qt graphics"""
        perms = self.eit_perms[permidx,:]
        # Clear existing items
        self.scene.clear()



        x = self.pts[:, 0]
        y = self.pts[:, 1]
        self.max_val = np.max(perms)
        self.min_val = np.min(perms)
        #print(self.max_val)
        #print(self.min_val)
        if self.max_val == self.min_val:
            return
        
        # Find data range for colormap
        #min_perm = np.min(perms)
        #max_perm = np.max(perms)
        
        # Draw each triangle
        for idx in range(0, self.tri.shape[0]):
            triangle = self.tri[idx,:]
            # Get vertices of the triangle
            triangle_points = []
            #perm_values = perms[idx]
            perm_values = []
            
            
            
            for vertex_idx in triangle:
                point = QPointF(x[vertex_idx], y[vertex_idx])
                triangle_points.append(point)
                perm_values.append(perms[vertex_idx])
            
            
            if len(perms) == self.tri.shape[0]: perm_values = perms[idx]
            # Create polygon for the triangle
            polygon = QPolygonF(triangle_points)
            avg_perm = np.mean(perm_values)
            
            
            # Create color based on z value
            color = self.create_colormap(avg_perm)
            #print(color)
            
            # Add triangle to scene
            self.scene.addPolygon(polygon, QPen(Qt.NoPen), QBrush(color))
    def calc_perms(self, eit_data):
        self.perm = self.eit.solve(eit_data[0,0:40], self.v0, normalize=True)
        #node_ds = sim2pts(self.pts, self.tri, np.real(self.perm))
        self.eit_perms = np.zeros((eit_data.shape[0], len(self.perm)))
        self.eit_perms[0,:] = self.perm
        self.averaged_data = self.running_average(eit_data, 4);
        
    
        for idx in range(5, eit_data.shape[0]):
            #d = self.eit_data[idx,0:40] - self.eit_data[idx-1,0:40]
            #perms = 192.0 * self.eit.solve(self.averaged_data[idx,0:40], self.averaged_data[0,0:40], normalize=True)
            #perms = np.linalg.lstsq(self.Jacobian, d, rcond=None)[0]
            perms = self.eit.solve(self.averaged_data[idx,0:40], self.averaged_data[5,0:40], normalize=True)
            #node_ds = sim2pts(self.pts, self.tri, np.real(perms))
            self.eit_perms[idx,:] = perms


        self.max_val = np.max(self.eit_perms[1:,:])
        self.min_val = np.min(self.eit_perms[1:,:])
        #print(self.max_val)
        #print(self.min_val)

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