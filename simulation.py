from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from random import random as rnd
from random import randint
import sys
import time
import numpy as np


class Simulator(QWidget):

    def __init__(self, width, height, drones_num, locations_num):
        super().__init__()
        self.width = width
        self.height = height
        self.w_scale = int((screen_w // width) * 0.75)
        self.h_scale = int((screen_h // height) * 0.75)
        self.drones_num = drones_num
        self.locations_num = locations_num

        # TODO: remove this mock value
        self.cycleNo = 0

        self.init_window()

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

    def get_trajectory(self, drone_index):
        # TODO: use the real trajectory
        return randint(0, locations_num-1)

    def updateCurrentCyle(self):
        self.tableWidget.item(3, 1).setText(str(self.currentCycle))

    def updateAvgAoI(self):
        self.tableWidget.item(4, 1).setText(str(self.avgAoI))

    def updatePeakAoI(self):
        self.tableWidget.item(5, 1).setText(str(self.peakAoI))

    def createLegend(self):
        self.tableWidget = QTableWidget()
        self.tableWidget.setFixedHeight(186)
        self.tableWidget.setFixedWidth(210)
        self.tableWidget.setColumnWidth(0, 500)
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.setShowGrid(False)
        self.tableWidget.setStyleSheet("QTableWidget {background-color: gray;}")

        label = ["Grid size (mt)", "Drones", "Locations", "Cycle", "Average AoI", "Peak AoI"]

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
        self.tableWidget.item(1, 1).setText(str(self.drones_num))
        self.tableWidget.item(2, 1).setText(str(self.locations_num))
        self.tableWidget.item(3, 1).setText(str(self.currentCycle))

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

        self.drones_locations = (np.random.rand(
            self.drones_num, 2) * map_size) - (map_size / 2)

        for i, (x, y) in enumerate(self.drones_locations):
            target_x, target_y = sensing_locations[self.get_trajectory(i)]

            painter.drawPixmap(int(center_x + x * self.w_scale),
                               int(center_y + y * self.h_scale), uav_size, uav_size, uav_pixmap)
            painter.drawLine(int(center_x + x * self.w_scale + uav_size / 2),
                             int(center_y + y * self.h_scale + uav_size / 2),
                             int(center_x + target_x * self.w_scale + location_size / 2),
                             int(center_y + target_y * self.h_scale + location_size / 2))

        for x, y in sensing_locations:
            painter.drawPixmap(int(center_x + x * self.w_scale),
                               int(center_y + y * self.h_scale),
                               location_size, location_size, locations_pixmap)

    def _update(self):
        self.updateCurrentCyle()
        self.updateAvgAoI()
        self.updatePeakAoI()
        # TODO: remove this line and use the real value
        self.cycleNo += 1
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


map_size = 100.0
locations_num = 30
drones_num = 10
sensing_locations = (np.random.rand(locations_num, 2) * map_size) - (map_size / 2)

app = QApplication(sys.argv)
screen_w, screen_h = get_screen_resolution(app)
simulator = Simulator(map_size, map_size, drones_num, locations_num)
simulator.show()
sys.exit(app.exec())
