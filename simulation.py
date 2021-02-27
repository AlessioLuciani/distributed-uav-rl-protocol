from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from random import random as rnd
from random import randint
import sys
import time
import numpy as np
from fire import Fire
from argparse import Namespace

class Simulator(QWidget):

    def __init__(self, screen_w, screen_h):
        super().__init__()
        self.w_scale = int((screen_w // self.width) * 0.75)
        self.h_scale = int((screen_h // self.height) * 0.75)
        self.is_paused = False
        self.drones_locations = (np.random.rand(options.drones_amount, 2) * self.width) - (self.width / 2)
        self.focus_drone = -1
        self.cycle_phases = ["Decision", "Empty", "Sensing", "Sending"]

        self.bs_size = 128
        self.location_size = 36
        self.uav_size = 34

        # TODO: remove this mock value
        self.cycleNo = 0

        self.init_window()
    
    def mousePressEvent(self, event):
        center_x = int(self.width * self.w_scale / 2)
        center_y = int(self.height * self.h_scale / 2)
        for i, (x,y), in enumerate(self.drones_locations):
            xx = int(center_x + x * self.w_scale)
            yy = int(center_y + y * self.h_scale)
            if xx - self.uav_size <= event.x() <= xx + self.uav_size and yy - self.uav_size <= event.y() <= yy + self.uav_size:
                self.focus_drone = i
                break
        self.updateFocusDrone()

    @property
    def width(self):
        return options.grid_size

    @property
    def height(self):
        return options.grid_size

    @property
    def currentCycle(self):
        # TODO: use the current cycle
        return self.cycleNo

    @property
    def avgAoI(self):
        # TODO: use the real avgAoI
        return round(rnd() * 42, 5)

    @property
    def peakAoI(self):
        # TODO: use the real peakAoI
        return round(42 + rnd() * 21, 5)

    @property
    def focused_drone_phase(self):
        # TODO: return actual drone phase using self.focus_drone
        return self.cycle_phases[0]

    def get_trajectory(self, drone_index):
        # TODO: use the real trajectory
        return randint(0, options.sensing_locations_amount-1)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            self.is_paused = not self.is_paused
        elif event.key() == Qt.Key_Backspace:
            self.cycleNo = max(0, self.cycleNo-1)

    def updateCurrentCyle(self):
        self.tableWidget.item(3, 1).setText(str(self.currentCycle))

    def updateAvgAoI(self):
        self.tableWidget.item(4, 1).setText(str(self.avgAoI))

    def updatePeakAoI(self):
        self.tableWidget.item(5, 1).setText(str(self.peakAoI))

    def updateFocusDrone(self):
        self.tableWidget.item(6, 1).setText(str(self.focus_drone))

    def createLegend(self):
        self.tableWidget = QTableWidget()
        self.tableWidget.setFixedHeight(242)
        self.tableWidget.setFixedWidth(210)
        self.tableWidget.setColumnWidth(0, 500)
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.setShowGrid(False)
        self.tableWidget.setStyleSheet("QTableWidget {background-color: gray;}")

        label = ["Grid size (mt)", "Drones", "Locations", "Cycle", "Average AoI", "Peak AoI", "Drone ID" , "Cycle phase"]

        self.tableWidget.setRowCount(len(label))
        self.tableWidget.setColumnCount(2)

        for i, label in enumerate(label):
            item = QTableWidgetItem(label)
            value = QTableWidgetItem("0")

            item.setFlags(item.flags() ^ (Qt.ItemIsEditable | Qt.ItemIsSelectable))
            value.setFlags(value.flags() ^ (Qt.ItemIsEditable | Qt.ItemIsSelectable))
            self.tableWidget.setItem(i, 0, item)
            self.tableWidget.setItem(i, 1, value)

        self.tableWidget.item(0, 1).setText(
            "{}x{}".format(int(self.width), int(self.height)))
        self.tableWidget.item(1, 1).setText(str(options.drones_amount))
        self.tableWidget.item(2, 1).setText(str(options.sensing_locations_amount))
        self.tableWidget.item(3, 1).setText(str(self.currentCycle))
        self.tableWidget.item(6, 1).setText(str(self.focus_drone))
        self.tableWidget.item(7, 1).setText(self.focused_drone_phase)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignRight | Qt.AlignTop)
        layout.addWidget(self.tableWidget)
        self.setLayout(layout)

    def paintEvent(self, event):

        painter = QPainter(self)
        painter.setPen(QPen(Qt.black, 1, Qt.DashLine))

        bs_size = 128
        location_size = 36
        uav_size = 34

        broadcast_pixmap = QPixmap("images/antenna.png")
        locations_pixmap = QPixmap("images/wifi.png")
        uav_pixmap = QPixmap("images/drone.png")

        center_x = int(self.width * self.w_scale / 2)
        center_y = int(self.height * self.h_scale / 2)
        painter.drawPixmap(int(center_x - bs_size / 2),
                           int(center_y - bs_size / 2),
                           bs_size, bs_size, broadcast_pixmap)


        for i, (x, y) in enumerate(self.drones_locations):
            target_x, target_y = options.sensing_locations[self.get_trajectory(i)]

            painter.drawPixmap(int(center_x + x * self.w_scale),
                               int(center_y + y * self.h_scale), uav_size, uav_size, uav_pixmap)
            painter.drawLine(int(center_x + x * self.w_scale + uav_size / 2),
                             int(center_y + y * self.h_scale + uav_size / 2),
                             int(center_x + target_x * self.w_scale + location_size / 2),
                             int(center_y + target_y * self.h_scale + location_size / 2))

        for x, y in options.sensing_locations:
            painter.drawPixmap(int(center_x + x * self.w_scale),
                               int(center_y + y * self.h_scale),
                               location_size, location_size, locations_pixmap)

    def _update(self):
        self.updateCurrentCyle()
        self.updateAvgAoI()
        self.updatePeakAoI()
        # TODO: remove this line and use the real value
        if not self.is_paused: self.cycleNo += 1
        self.update()

    def init_window(self):
        self.setWindowTitle("Distributed UAV-RL simulator")
        self.setFixedSize(int(self.width * self.w_scale),
                          int(self.height * self.h_scale))
        self.setGeometry(0, 0, int(self.width * self.w_scale), int(self.height * self.h_scale))

        pal = self.palette()
        pal.setColor(QPalette.Background, Qt.white)
        self.setAutoFillBackground(True)
        self.setPalette(pal)

        # TODO: check if the periodic update every second is needed
        self.qTimer = QTimer()
        self.qTimer.setInterval(1000)
        self.qTimer.timeout.connect(self._update)
        self.qTimer.start()

        self.createLegend()

def get_screen_resolution(app):
    screen = app.primaryScreen()
    screen_size = screen.size()
    return screen_size.width(), screen_size.height()

options = Namespace(
    sensing_locations = [],
    sensing_locations_amount = 10,
    grid_size=100.0,
    drones_amount=5
)

def main(grid_size=options.grid_size, sensing_locations_amount=options.sensing_locations_amount, drones_amount=options.drones_amount):
    sensing_locations = (np.random.rand(sensing_locations_amount, 2) * grid_size) - (grid_size / 2)
    options.sensing_locations = sensing_locations
    options.sensing_locations_amount = sensing_locations_amount
    options.grid_size = grid_size
    options.drones_amount = drones_amount

    app = QApplication(sys.argv)
    screen_w, screen_h = get_screen_resolution(app)
    simulator = Simulator(screen_w, screen_h)
    simulator.show()
    sys.exit(app.exec())

if __name__ == '__main__':
  Fire(main)
