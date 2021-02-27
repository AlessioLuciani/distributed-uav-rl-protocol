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
from random import random as rnd
from fire import Fire
from argparse import Namespace
import sys, math, time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



class Simulator(QWidget):

    def __init__(self, screen_w, screen_h):
        super().__init__()
        self.w_scale = int((screen_w // self.width) * 0.75)
        self.h_scale = int((screen_h // self.height) * 0.75)
        self.cycle_num = 0
        self.init_window()

    @property
    def width(self):
        return options.grid_size

    @property
    def height(self):
        return options.grid_size

    @property
    def currentCycle(self):
        return self.cycle_num

    @property
    def avgAoI(self):
        return np.mean(options.aois)

    @property
    def peakAoI(self):
        return np.max(options.aois)

    def get_trajectory(self, drone_index):
        return options.chosen_locations[drone_index]

    def updateCurrentCyle(self):
        self.tableWidget.item(3, 1).setText("{}/{}".format(self.currentCycle, options.cycles_num))

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
        self.tableWidget.item(1, 1).setText(str(options.drones_amount))
        self.tableWidget.item(2, 1).setText(str(options.sensing_locations_amount))
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

        for i, (x, y) in enumerate(options.drones_locations):
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
    return (cycle * options.cycle_length) - ((options.aois[location_index] // options.cycle_length) * options.cycle_length)

def get_accumulated_aoi(cycle):
    return max([sum([get_location_aoi(cycle, location) for location in range(options.sensing_locations_amount)]), 0.0001])

def get_trajectory(drone_index):
    chosen_location_index = options.chosen_locations[drone_index]
    chosen_location = options.sensing_locations[chosen_location_index]
    drone_location = options.drones_locations[drone_index]
    distance = np.linalg.norm(chosen_location - drone_location)
    if distance <= options.drone_max_speed * options.cycle_length:
        return chosen_location - drone_location
    else:
        return ((chosen_location - drone_location) / distance) * options.drone_max_speed * options.cycle_length

options = Namespace(
    cycle_length = 1.0,
    sensing_locations = np.array([]),
    aois = np.array([]),
    sensing_data_amounts = np.array([]),
    drones_locations = np.array([]),
    sensing_locations_amount = 10,
    cycle_stages = np.array([]),
    data_transmission_cycle = 0.0,
    grid_size = 100.0,
    drones_amount = 5,
    drone_max_speed = 5.0,
    drone_bandwidth = 0.5,
    total_location_data = 1.5,
    cycles_num = 1000
)

def main(grid_size=options.grid_size, sensing_locations_amount=options.sensing_locations_amount, drones_amount=options.drones_amount,
            drone_max_speed=options.drone_max_speed, drone_bandwidth=options.drone_bandwidth, total_location_data=options.total_location_data,
            cycles_num=options.cycles_num):
    sensing_locations = (np.random.rand(sensing_locations_amount, 2) * grid_size) - (grid_size / 2)
    options.sensing_locations = sensing_locations
    options.sensing_locations_amount = sensing_locations_amount
    options.grid_size = grid_size
    options.drones_amount = drones_amount
    options.drone_max_speed = drone_max_speed
    options.drone_bandwidth = drone_bandwidth
    options.cycles_num = cycles_num

    # -- BEGIN of the jupyter notebook's code (please refer to it for a more readable version)

    options.drones_locations = np.zeros((options.drones_amount, 2), dtype=float)
    options.sensing_data_amounts = np.zeros(options.drones_amount, dtype=float)
    options.aois = np.zeros(options.sensing_locations_amount, dtype=int)
    options.chosen_locations = np.zeros(options.drones_amount, dtype=int)
    options.cycle_stages = np.zeros(options.drones_amount, dtype=int)
    options.data_transmission_cycle = options.drone_bandwidth * options.cycle_length

    class DurpEnv(py_environment.PyEnvironment):
        def __init__(self, drone):
            self._drone = drone
            self.cycle = 1
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(1,), dtype=np.float64, minimum=0, maximum=options.sensing_locations_amount-0.001, name='action')
            self._observation_spec = array_spec.BoundedArraySpec(shape=(2,),
                minimum=(-options.grid_size/2), maximum=(options.grid_size/2), dtype=np.float64)
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
            options.aois[chosen_location_index] = self.cycle
            new_accumulated_aoi = get_accumulated_aoi(self.cycle)
            aoi_multiplier = 0.05
            normalized_diff_aoi_component = \
                ((1/(1+np.exp(- (accumulated_aoi - new_accumulated_aoi) * aoi_multiplier))) - 0.5) * 2.0
            chosen_location = options.sensing_locations[chosen_location_index]
            drone_location = options.drones_locations[self._drone]
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
    for drone in range(options.drones_amount):
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
    num_iterations = 20
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
        options.aois = np.zeros(options.sensing_locations_amount, dtype=int)
        collect_data(env, random_policy, replay_buffer, initial_collect_steps)
        options.aois = np.zeros(options.sensing_locations_amount, dtype=int)
        dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=batch_size, 
        num_steps=2,
        single_deterministic_pass=False).prefetch(3)
        iterator = iter(dataset)
        agent.train_step_counter.assign(0)
        for i in range(num_iterations):
            options.aois = np.zeros(options.sensing_locations_amount, dtype=int)
            collect_data(env, agent.collect_policy, replay_buffer, 1)
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss
            step = agent.train_step_counter.numpy()

    time_steps = []
    for drone in range(options.drones_amount):
        time_steps.append(environments[drone].reset())
    
    options.aois = np.zeros(options.sensing_locations_amount, dtype=int)

    # -- END of the jupyter notebook's code

    # -- Start of the simulation

    app = QApplication(sys.argv)
    screen_w, screen_h = get_screen_resolution(app)
    simulator = Simulator(screen_w, screen_h)
    simulator.show()

    for cycle in range(1, options.cycles_num + 1):
        if cycle % 100 == 1:
            print(get_accumulated_aoi(cycle))
            print(options.aois)
            print("----------------")
        for drone in range(options.drones_amount):
            if options.cycle_stages[drone] == 0:
                agent = agents[drone]
                env = environments[drone]
                agent.cycle = cycle
                policy_step = agent.policy.action(time_steps[drone]) 
                new_step = env.step(policy_step.action)
                time_steps[drone] = new_step
                options.chosen_locations[drone] = int(policy_step.action)
                options.cycle_stages[drone] =  1
            elif (options.drones_locations[drone] != options.sensing_locations[options.chosen_locations[drone]]).all():
                traj = get_trajectory(drone)
                new_location = options.drones_locations[drone] + traj
                options.drones_locations[drone] = new_location
            elif options.sensing_data_amounts[drone] == 0.0:
                options.cycle_stages[drone] =  2
                options.sensing_data_amounts[drone] = options.total_location_data
                options.cycle_stages[drone] =  3
            else:
                options.sensing_data_amounts[drone] = np.max([options.sensing_data_amounts[drone] - options.data_transmission_cycle, 0.0])
                if options.sensing_data_amounts[drone] == 0.0:
                    options.aois[options.chosen_locations[drone]] = cycle
                    options.cycle_stages[drone] =  0
        simulator._update()
        #time.sleep(1) # TODO: find a value that allows the user enjoy the simulation
    
    sys.exit(app.exec())

if __name__ == '__main__':
  Fire(main)
