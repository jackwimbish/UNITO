import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
import math

class DraggablePoint:
    def __init__(self, point, line, index, xdata, ydata):
        self.point = point
        self.line = line
        self.index = index
        self.xdata = xdata
        self.ydata = ydata
        self.press = None

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
    def __init__(self, num_points=5):
        super().__init__()

        self.num_points = num_points

        self.setWindowTitle("Matplotlib in PyQt5")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.plot()

    def plot(self):
        data = np.random.random((100, 100))
        self.canvas.axes.imshow(data, interpolation='none')

        # Generate points for a regular polygon (convex shape)
        angle_step = 2 * math.pi / self.num_points
        radius = 40
        center_x, center_y = 50, 50

        self.x = np.array([center_x + radius * math.cos(i * angle_step) for i in range(self.num_points)])
        self.y = np.array([center_y + radius * math.sin(i * angle_step) for i in range(self.num_points)])
        
        # Ensure the points form a closed loop
        self.x = np.append(self.x, self.x[0])
        self.y = np.append(self.y, self.y[0])

        self.line, = self.canvas.axes.plot(self.x, self.y, marker='o', color='r')

        self.points = []
        for i, (xi, yi) in enumerate(zip(self.x, self.y)):
            point = plt.Circle((xi, yi), 2, color='r', picker=True)
            self.canvas.axes.add_artist(point)
            dp = DraggablePoint(point, self.line, i, self.x, self.y)
            dp.connect()
            self.points.append(dp)

        self.canvas.draw()

app = QApplication(sys.argv)

# You can change the number of points here
num_points = 6
window = MainWindow(num_points=num_points)
window.show()
sys.exit(app.exec_())
