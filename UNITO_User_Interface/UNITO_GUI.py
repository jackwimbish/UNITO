import sys
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QComboBox
import gc
from scipy.spatial import ConvexHull
import datashader as ds
from datashader.mpl_ext import dsshow

# Import necessary modules from your custom scripts
from Data_Preprocessing import process_table, prepare_subj
from Predict import UNITO_gating, evaluation

warnings.filterwarnings("ignore")

# Setting gates
gate_pre_list = [None, 'Lymphocyte']
gate_list = ['Lymphocyte', 'Singlet']
x_axis_list = ['FSC_A', 'FSC_W']
y_axis_list = ['SSC_A', 'SSC_W']
path2_lastgate_pred_list = ['./Raw_Data/', './prediction/']

# Define model parameters
device = 'cpu'  # can change to 'cuda' or 'mps', but prediction with cpu is sufficient and works for all computers
n_worker = 0

# Define paths
dest = '.'  # change depending on your needs
save_data_img_path = f'{dest}/Data_image'
save_figure_path = f'{dest}/figures'
save_prediction_path = f'{dest}/prediction'

# Make directories
if not os.path.exists(save_data_img_path):
    os.mkdir(save_data_img_path)
if not os.path.exists(save_figure_path):
    os.mkdir(save_figure_path)
if not os.path.exists(save_prediction_path):
    os.mkdir(save_prediction_path)

class DraggablePoint:
    def __init__(self, point, line, index, xdata, ydata, ax):
        self.point = point
        self.line = line
        self.index = index
        self.xdata = xdata
        self.ydata = ydata
        self.press = None
        self.ax = ax
        self.connect()

    def connect(self):
        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.point.axes: return
        contains, _ = self.point.contains(event)
        if not contains: return
        self.press = (self.point.center), event.xdata, event.ydata

    def on_motion(self, event):
        if self.press is None: return
        if event.inaxes != self.point.axes: return
        center, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        new_center = (center[0] + dx, center[1] + dy)
        self.point.set_center(new_center)
        self.update_line(new_center)
        self.point.figure.canvas.draw()

    def on_release(self, event):
        self.press = None
        self.point.figure.canvas.draw()

    def update_line(self, new_center):
        self.xdata[self.index] = new_center[0]
        self.ydata[self.index] = new_center[1]
        self.line.set_data(self.xdata, self.ydata)

    def disconnect(self):
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Data Analysis Tool")
        self.setGeometry(100, 100, 1200, 600)

        # Layouts
        main_layout = QHBoxLayout()
        button_layout = QVBoxLayout()

        # Buttons
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        button_layout.addWidget(self.load_button)

        self.gate_combobox = QComboBox()
        self.gate_combobox.addItems(gate_list)
        self.gate_combobox.setEnabled(False)
        button_layout.addWidget(self.gate_combobox)

        self.confirm_button = QPushButton("Confirm Selection")
        self.confirm_button.clicked.connect(self.on_confirm_selection)
        self.confirm_button.setEnabled(False)
        button_layout.addWidget(self.confirm_button)

        self.predict_button = QPushButton("Prediction")
        self.predict_button.clicked.connect(self.make_prediction)
        button_layout.addWidget(self.predict_button)

        self.visualize_button = QPushButton("Visualization")
        self.visualize_button.clicked.connect(self.visualize_data)
        button_layout.addWidget(self.visualize_button)

        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self.save_plot)
        button_layout.addWidget(self.save_button)

        # Placeholder for filename
        self.filename_label = QLabel("No file loaded")
        button_layout.addWidget(self.filename_label)

        # Canvas for Matplotlib
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)

        # Add layouts to main layout
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.canvas)

        self.setLayout(main_layout)

        # Variables for selected gate and parameters
        self.gate = None
        self.gate_pre = None
        self.x_axis = None
        self.y_axis = None
        self.path_raw = None

        # Global variable to store file name
        self.file_name = None

    def load_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            base_name = file_name.split('/')[-1]
            self.file_name = base_name
            self.filename_label.setText(base_name)
            self.gate_combobox.setEnabled(True)
            self.confirm_button.setEnabled(True)
            print(f"Selected file: {base_name}")  # For debugging, you can remove this later

    def on_confirm_selection(self):
        index = self.gate_combobox.currentIndex()
        if index >= 0:
            self.gate = gate_list[index]
            self.gate_pre = gate_pre_list[index]
            self.x_axis = x_axis_list[index]
            self.y_axis = y_axis_list[index]
            self.path_raw = path2_lastgate_pred_list[index]

            # Run the provided functions
            process_table(self.x_axis, self.y_axis, self.gate_pre, self.gate, self.file_name, seq=(self.gate_pre != None))
            prepare_subj(self.gate)

            print(f"Selected Gate: {self.gate}")
            print(f"Gate Pre: {self.gate_pre}")
            print(f"X Axis: {self.x_axis}")
            print(f"Y Axis: {self.y_axis}")
            print(f"Path Raw: {self.path_raw}")

    def make_prediction(self):
        if self.gate and self.x_axis and self.y_axis:
            model_path = f'./Model/{self.gate}_model.pt'
            data_df_pred = UNITO_gating(model_path, self.x_axis, self.y_axis, self.gate, self.path_raw, n_worker, device, save_prediction_path, seq=(self.gate_pre != None), gate_pre=self.gate_pre)
            print("Predictions made.")
        else:
            print("No data loaded or gate not selected.")

    def visualize_data(self):
        if self.gate and self.x_axis and self.y_axis and self.path_raw:
            self.canvas.axes.clear()
            data_table = pd.read_csv(f'./prediction/{self.file_name}')

            x_axis_pred = self.x_axis
            y_axis_pred = self.y_axis

            if self.gate_pre != None:
                gate1_pred = self.gate_pre + '_pred'
            gate2_pred = self.gate + '_pred'

            data = data_table.copy()
            if self.gate_pre != None:
                data = data[data[gate1_pred] == 1]
            in_gate = data[data[gate2_pred] == 1]
            out_gate = data[data[gate2_pred] == 0]
            in_gate = in_gate[[x_axis_pred, y_axis_pred]].to_numpy()
            out_gate = out_gate[[x_axis_pred, y_axis_pred]].to_numpy()

            if in_gate.shape[0] > 3:
                hull = ConvexHull(in_gate)
                line, = self.canvas.axes.plot(
                    np.append(in_gate[hull.vertices, 0], in_gate[hull.vertices[0], 0]), 
                    np.append(in_gate[hull.vertices, 1], in_gate[hull.vertices[0], 1]), 
                    'k-'
                )

                # Add draggable points

                self.points = []
                for i, (xi, yi) in enumerate(in_gate[hull.vertices]):
                    point = plt.Circle((xi, yi), 0.5, color='r', picker=True)
                    self.canvas.axes.add_artist(point)
                    dp = DraggablePoint(point, line, i, in_gate[hull.vertices, 0], in_gate[hull.vertices, 1], self.canvas.axes)
                    dp.connect()
                    self.points.append(dp)

                density_plot = dsshow(
                    data,
                    ds.Point(x_axis_pred, y_axis_pred),
                    ds.count(),
                    vmin=0,
                    vmax=300,
                    norm='linear',
                    cmap='jet',
                    aspect='auto',
                    ax=self.canvas.axes,
                )

            self.canvas.axes.set_xlabel(self.x_axis)
            self.canvas.axes.set_ylabel(self.y_axis)
            self.canvas.axes.set_title("UNITO Auto-gating")

            self.canvas.draw()
        else:
            print("No data or gate not selected")

    def save_plot(self):
        if self.gate and self.x_axis and self.y_axis and self.path_raw:
            subject = self.file_name.split('/')[-1]
            data_table = pd.read_csv(f'./prediction/{subject}')
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))

            ###########################################################
            # True
            ###########################################################

            data = data_table.copy()
            if self.gate_pre != None:
                data = data[data[self.gate_pre]==1]
            
            in_gate = data[data[self.gate] == 1]
            out_gate = data[data[self.gate] == 0]
            in_gate = in_gate[[self.x_axis, self.y_axis]].to_numpy()
            out_gate = out_gate[[self.x_axis, self.y_axis]].to_numpy()

            if in_gate.shape[0] > 3:
                hull = ConvexHull(in_gate)
                for simplex in hull.simplices:
                    ax[0].plot(in_gate[simplex, 0], in_gate[simplex, 1], 'k-')

                density_plot = dsshow(
                    data,
                    ds.Point(self.x_axis, self.y_axis),
                    ds.count(),
                    vmin=0,
                    vmax=300,
                    norm='linear',
                    cmap='jet',
                    aspect='auto',
                    ax=ax[0],
                )

            # cbar = plt.colorbar(density_plot)
            # cbar.set_label('Number of cells in pixel')
            ax[0].set_xlabel(self.x_axis)
            ax[0].set_ylabel(self.y_axis)

            ax[0].set_title("Manual Gating")


            ###########################################################
            # predict
            ###########################################################
            
            x_axis_pred = self.x_axis
            y_axis_pred = self.y_axis

            if self.gate_pre != None:
                gate1_pred = self.gate_pre + '_pred'
            gate2_pred = self.gate + '_pred'

            data_table = pd.read_csv(f'./prediction/{subject}')

            data = data_table.copy()
            if self.gate_pre != None:
                data = data[data[gate1_pred]==1]
            in_gate = data[data[gate2_pred] == 1]
            out_gate = data[data[gate2_pred] == 0]
            in_gate = in_gate[[x_axis_pred, y_axis_pred]].to_numpy()
            out_gate = out_gate[[x_axis_pred, y_axis_pred]].to_numpy()

            if in_gate.shape[0] > 3:
                hull = ConvexHull(in_gate)
                for simplex in hull.simplices:
                    ax[1].plot(in_gate[simplex, 0], in_gate[simplex, 1], 'k-')

                density_plot = dsshow(
                    data,
                    ds.Point(x_axis_pred, y_axis_pred),
                    ds.count(),
                    vmin=0,
                    vmax=300,
                    norm='linear',
                    cmap='jet',
                    aspect='auto',
                    ax=ax[1],
                )

            # cbar = plt.colorbar(density_plot)
            # cbar.set_label('Number of cells in pixel')
            ax[1].set_xlabel(self.x_axis)
            ax[1].set_ylabel(self.y_axis)

            ax[1].set_title("UNITO Auto-gating")

            # save figure
            plt.savefig(f'./figures/Figure_{self.gate}_Recon_Sequential_{subject}.png')
            plt.close()
        else:
            print("No data or gate not selected")

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
