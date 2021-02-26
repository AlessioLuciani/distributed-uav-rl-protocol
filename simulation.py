


from tf_agents.networks import q_network
from tf_agents.agents.ddpg import ddpg_agent, critic_network, actor_network, critic_rnn_network, actor_rnn_network
from tf_agents.specs import tensor_spec, array_spec
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.policies import random_tf_policy
from tf_agents import utils
from tf_agents.trajectories import time_step as ts
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from random import randint
import sys, math, time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from random import random as rnd


class Simulator(QWidget):

    def __init__(self, width, height, drones_num, locations_num):
        super().__init__()
        self.width = width
        self.height = height
        self.w_scale = int((screen_w // width) * 0.75)
        self.h_scale = int((screen_h // height) * 0.75)
        self.drones_num = drones_num
        self.locations_num = locations_num
        self.cycle_num = 0

        self.init_window()

    @property
    def currentCycle(self):
        return self.cycle_num

    @property
    def avgAoI(self):
        return np.mean(aois)

    @property
    def peakAoI(self):
        return np.max(aois)

    def get_trajectory(self, drone_index):
        return chosen_locations[drone_index]

    def updateCurrentCyle(self):
        self.tableWidget.item(3, 1).setText("{}/{}".format(self.currentCycle, cycles_num))

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

        for i, (x, y) in enumerate(drones_locations):
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
        self.cycle_num += 1
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

        self.createLegend()

def get_screen_resolution(app):
    screen = app.primaryScreen()
    screen_size = screen.size()
    return screen_size.width(), screen_size.height()

def get_location_aoi(cycle, location_index):
    return (cycle * cycle_length) - ((aois[location_index] // cycle_length) * cycle_length)

def get_accumulated_aoi(cycle):
    return max([sum([get_location_aoi(cycle, location) for location in range(locations_num)]), 0.0001])

def get_trajectory(drone_index):
    chosen_location_index = chosen_locations[drone_index]
    chosen_location = sensing_locations[chosen_location_index]
    drone_location = drones_locations[drone_index]
    distance = np.linalg.norm(chosen_location - drone_location)
    if distance <= max_speed * cycle_length:
        return chosen_location - drone_location
    else:
        return ((chosen_location - drone_location) / distance) * max_speed * cycle_length

# -- BEGIN of the jupyter notebook's code (please refer to it for a more readable version)

map_size = 100.0
locations_num = 10
drones_num = 5
sensing_locations = (np.random.rand(locations_num, 2) * map_size) - (map_size / 2)
cycle_length = 1.0
max_speed = 5.0
drones_locations = np.zeros((drones_num, 2), dtype=float)
sensing_data_amounts = np.zeros(drones_num, dtype=float)
aois = np.zeros(locations_num, dtype=int)
chosen_locations = np.zeros(drones_num, dtype=int)
cycle_stages = np.zeros(drones_num, dtype=int)
bandwidth = 0.5
total_location_data = 1.5
data_transmission_cycle = bandwidth * cycle_length

class DurpEnv(py_environment.PyEnvironment):
    def __init__(self, drone):
        self._drone = drone
        self.cycle = 1
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.float64, minimum=0, maximum=locations_num-0.001, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(2,),
            minimum=(-map_size/2), maximum=(map_size/2), dtype=np.float64)
        self._state = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        return ts.restart(np.array([0.0,0.0], dtype=np.float64))

    def _step(self, action):
        chosen_location_index = int(action)
        accumulated_aoi = get_accumulated_aoi(self.cycle)
        aois[chosen_location_index] = self.cycle
        new_accumulated_aoi = get_accumulated_aoi(self.cycle)
        aoi_multiplier = 0.05
        normalized_diff_aoi_component = \
            ((1/(1+np.exp(- (accumulated_aoi - new_accumulated_aoi) * aoi_multiplier))) - 0.5) * 2.0
        chosen_location = sensing_locations[chosen_location_index]
        drone_location = drones_locations[self._drone]
        distance = np.linalg.norm(chosen_location - drone_location)
        distance_multiplier = 0.03
        normalized_location_distance = (1 - (1/(1+np.exp(-distance * distance_multiplier)))) * 2.0
        aoi_weight = 0.99
        reward = normalized_diff_aoi_component * aoi_weight + normalized_location_distance * (1.0 - aoi_weight)
        self.cycle += 8
        return ts.transition(chosen_location, reward=reward)

learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
environments = []
agents = []
for drone in range(drones_num):
    durp_env = DurpEnv(drone)
    train_env = tf_py_environment.TFPyEnvironment(durp_env)
    actor_rnn_net = actor_rnn_network.ActorRnnNetwork(
        train_env.observation_spec(), 
        train_env.action_spec()
    )
    critic_rnn_net = critic_rnn_network.CriticRnnNetwork(
        (train_env.observation_spec(), train_env.action_spec()), 
        lstm_size=[2]
    )
    actor_net = actor_network.ActorNetwork(
        train_env.observation_spec(), 
        train_env.action_spec()
    )
    critic_net = critic_network.CriticNetwork(
        (train_env.observation_spec(), train_env.action_spec()), 
        output_activation_fn=tf.keras.activations.sigmoid,
        activation_fn=tf.nn.relu
    )
    agent = ddpg_agent.DdpgAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    critic_network=critic_net,
    actor_network=actor_net,
    critic_optimizer=optimizer,
    actor_optimizer=optimizer,
    )
    agent.initialize()
    agents.append(agent)
    environments.append(train_env)
num_iterations = 500
initial_collect_steps = 100
collect_steps_per_iteration = 100
batch_size = 64

def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

for i in range(len(agents)):
    agent = agents[i]
    env = environments[i]
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=env.batch_size)
    random_policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(), env.action_spec())              
    aois = np.zeros(locations_num, dtype=int)
    collect_data(env, random_policy, replay_buffer, initial_collect_steps)
    aois = np.zeros(locations_num, dtype=int)
    dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2,
    single_deterministic_pass=False).prefetch(3)
    iterator = iter(dataset)
    agent.train_step_counter.assign(0)
    for i in range(num_iterations):
        aois = np.zeros(locations_num, dtype=int)
        collect_data(env, agent.collect_policy, replay_buffer, 1)
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss
        step = agent.train_step_counter.numpy()

time_steps = []
for drone in range(drones_num):
    time_steps.append(environments[drone].reset())
cycles_num = 1000
aois = np.zeros(locations_num, dtype=int)

# -- END of the jupyter notebook's code

# -- Start of the simulation

app = QApplication(sys.argv)
screen_w, screen_h = get_screen_resolution(app)
simulator = Simulator(map_size, map_size, drones_num, locations_num)
simulator.show()

for cycle in range(1, cycles_num + 1):
    if cycle % 100 == 1:
        print(get_accumulated_aoi(cycle))
        print(aois)
        print("----------------")
    for drone in range(drones_num):
        if cycle_stages[drone] == 0:
            agent = agents[drone]
            env = environments[drone]
            agent.cycle = cycle
            policy_step = agent.policy.action(time_steps[drone]) 
            new_step = env.step(policy_step.action)
            time_steps[drone] = new_step
            chosen_locations[drone] = int(policy_step.action)
            cycle_stages[drone] =  1
        elif (drones_locations[drone] != sensing_locations[chosen_locations[drone]]).all():
            traj = get_trajectory(drone)
            new_location = drones_locations[drone] + traj
            drones_locations[drone] = new_location
        elif sensing_data_amounts[drone] == 0.0:
            cycle_stages[drone] =  2
            sensing_data_amounts[drone] = total_location_data
            cycle_stages[drone] =  3
        else:
            sensing_data_amounts[drone] = np.max([sensing_data_amounts[drone] - data_transmission_cycle, 0.0])
            if sensing_data_amounts[drone] == 0.0:
                aois[chosen_locations[drone]] = cycle
                cycle_stages[drone] =  0
    simulator._update()
    #time.sleep(1) # TODO: find a value that allows the user enjoy the simulation

sys.exit(app.exec())
