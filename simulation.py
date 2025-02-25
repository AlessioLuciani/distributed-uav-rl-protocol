#!/usr/bin/env python3

from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ddpg import (
    ddpg_agent,
    critic_network,
    actor_network,
    critic_rnn_network,
    actor_rnn_network,
)
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
from random import randint, randrange
from random import random as rnd
from fire import Fire
from argparse import Namespace
import sys
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class Simulator(QWidget):
    def __init__(self, screen_w, screen_h):
        super().__init__()
        self.w_scale = int((screen_w // self.width) * 0.75)
        self.h_scale = int((screen_h // self.height) * 0.75)
        self.cycle_num = 0
        self.is_paused = False
        self.focus_drone = -1
        self.focus_drone_phase = -1
        self.cycle_phases = ["Decision", "Empty", "Sensing", "Sending"]

        self.bs_size = 128
        self.location_size = 36
        self.uav_size = 34
        self.init_window()

    def mousePressEvent(self, event):
        center_x = int(self.width * self.w_scale / 2)
        center_y = int(self.height * self.h_scale / 2)
        for i, (x, y), in enumerate(options.drones_vec[self.currentCycle - 1]):
            xx = int(center_x + x * self.w_scale)
            yy = int(center_y + y * self.h_scale)
            if (
                xx - self.uav_size <= event.x() <= xx + self.uav_size
                and yy - self.uav_size <= event.y() <= yy + self.uav_size
            ):
                self.focus_drone = i
                break
        self.updateFocusDrone()
        self.updateFocusIcon()

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
        return np.mean(options.aois_vec[self.currentCycle - 1])

    @property
    def peakAoI(self):
        return np.max(options.aois_vec[self.currentCycle - 1])

    @property
    def focused_drone_phase(self):
        if self.currentCycle <= 0 or self.focus_drone < 0:
            return ""
        cycle_phase_idx = options.cycle_stages_vec[self.currentCycle - 1][
            self.focus_drone
        ]
        return self.cycle_phases[cycle_phase_idx]

    def get_trajectory(self, drone_index):
        return options.chosen_loc_vec[self.currentCycle - 1][drone_index]

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            self.is_paused = not self.is_paused
        elif event.key() == Qt.Key_Backspace:
            self.cycle_num = max(1, self.cycle_num - 1)

    def updateCurrentCyle(self):
        self.tableWidget.item(3, 1).setText(
            "{}/{}".format(self.currentCycle + 1, options.cycles_num)
        )

    def updateAvgAoI(self):
        self.tableWidget.item(4, 1).setText(str(self.avgAoI))

    def updatePeakAoI(self):
        self.tableWidget.item(5, 1).setText(str(self.peakAoI))

    def updateFocusDrone(self):
        self.tableWidget.item(6, 1).setText(str(self.focus_drone))

    def updateFocusIcon(self):
        self.tableWidget.item(7, 1).setText(self.focused_drone_phase)

    def createLegend(self):
        self.tableWidget = QTableWidget()
        self.tableWidget.setFixedHeight(242)
        self.tableWidget.setFixedWidth(210)
        self.tableWidget.setColumnWidth(0, 500)
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.setShowGrid(False)
        self.tableWidget.setStyleSheet(
            "QTableWidget {background-color: gray;}")

        label = [
            "Grid size (mt)",
            "Drones",
            "Locations",
            "Cycle",
            "Average AoI",
            "Peak AoI",
            "Drone ID",
            "Cycle stage",
        ]

        self.tableWidget.setRowCount(len(label))
        self.tableWidget.setColumnCount(2)

        for i, label in enumerate(label):
            item = QTableWidgetItem(label)
            value = QTableWidgetItem("0")

            item.setFlags(item.flags() ^ (
                Qt.ItemIsEditable | Qt.ItemIsSelectable))
            value.setFlags(value.flags() ^ (
                Qt.ItemIsEditable | Qt.ItemIsSelectable))
            self.tableWidget.setItem(i, 0, item)
            self.tableWidget.setItem(i, 1, value)

        self.tableWidget.item(0, 1).setText(
            "{}x{}".format(int(self.width), int(self.height))
        )
        self.tableWidget.item(1, 1).setText(str(options.drones_amount))
        self.tableWidget.item(2, 1).setText(
            str(options.sensing_locations_amount))
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

        broadcast_pixmap = QPixmap("images/antenna.png")
        locations_pixmap = QPixmap("images/wifi.png")
        uav_pixmap = QPixmap("images/drone.png")

        center_x = int(self.width * self.w_scale / 2)
        center_y = int(self.height * self.h_scale / 2)
        painter.drawPixmap(
            int(center_x - self.bs_size / 2),
            int(center_y - self.bs_size / 2),
            self.bs_size,
            self.bs_size,
            broadcast_pixmap,
        )

        for i, (x, y) in enumerate(options.drones_vec[self.currentCycle - 1]):
            target_x, target_y = options.sensing_locations[self.get_trajectory(
                i)]

            painter.drawPixmap(
                int(center_x + x * self.w_scale),
                int(center_y + y * self.h_scale),
                self.uav_size,
                self.uav_size,
                uav_pixmap,
            )
            if self.focus_drone < 0 or self.focus_drone != i:

                painter.setPen(QPen(Qt.black, 1, Qt.DashLine))
            else:
                painter.setPen(QPen(Qt.red, 1, Qt.DashLine))

            painter.drawLine(
                int(center_x + x * self.w_scale + self.uav_size / 2),
                int(center_y + y * self.h_scale + self.uav_size / 2),
                int(center_x + target_x * self.w_scale + self.location_size / 2),
                int(center_y + target_y * self.h_scale + self.location_size / 2),
            )

        for x, y in options.sensing_locations:
            painter.drawPixmap(
                int(center_x + x * self.w_scale),
                int(center_y + y * self.h_scale),
                self.location_size,
                self.location_size,
                locations_pixmap,
            )
        if self.currentCycle == options.cycles_num:
            self.qTimer.stop()
            self.close()

    def _update(self):
        self.updateFocusIcon()
        self.updateCurrentCyle()
        self.updateAvgAoI()
        self.updatePeakAoI()
        self.update()
        if not self.is_paused:
            self.cycle_num += 1

    def init_window(self):
        self.setWindowTitle("Distributed UAV-RL simulator")
        self.setFixedSize(
            int(self.width * self.w_scale), int(self.height * self.h_scale)
        )
        self.setGeometry(
            0, 0, int(self.width * self.w_scale), int(self.height * self.h_scale)
        )

        pal = self.palette()
        pal.setColor(QPalette.Background, Qt.white)
        self.setAutoFillBackground(True)
        self.setPalette(pal)

        self.qTimer = QTimer()
        self.qTimer.setInterval(500)
        self.qTimer.timeout.connect(self._update)
        self.qTimer.start()

        self.createLegend()


def get_screen_resolution(app):
    screen = app.primaryScreen()
    screen_size = screen.size()
    return screen_size.width(), screen_size.height()


def get_location_aoi(cycle, location_index):
    return (cycle * options.cycle_length) - (
        (options.aois[location_index] //
         options.cycle_length) * options.cycle_length
    )


def get_accumulated_aoi(cycle):
    return max(
        [
            sum(
                [
                    get_location_aoi(cycle, location)
                    for location in range(options.sensing_locations_amount)
                ]
            ),
            0.0001,
        ]
    )


def get_trajectory(drone_index):
    chosen_location_index = options.chosen_locations[drone_index]
    chosen_location = options.sensing_locations[chosen_location_index]
    drone_location = options.drones_locations[drone_index]
    distance = np.linalg.norm(chosen_location - drone_location)
    if distance <= options.drone_max_speed * options.cycle_length:
        return chosen_location - drone_location
    else:
        return (
            ((chosen_location - drone_location) / distance)
            * options.drone_max_speed
            * options.cycle_length
        )


def reset_aois():
    for i in range(options.sensing_locations_amount):
        options.aois[i] = 0


def reset_drones_locations():
    for i in range(options.drones_amount):
        options.drones_locations[i] = np.array([0.0, 0.0])


def get_best_action(drone):
    highest_reward = 0
    best_action = 0

    for action in range(options.sensing_locations_amount):

        chosen_location_index = action
        aoi = get_location_aoi(options.current_cycle[0], chosen_location_index)
        aoi_multiplier = 0.05

        # Using sigmoid function to obtain a value between 0 and 1. The
        # higher the aoi difference the better, so the higher
        # the reward.
        normalized_diff_aoi_component = (
            (1 / (1 + np.exp(-aoi * aoi_multiplier))) - 0.5
        ) * 2.0

        # Taking distance between current location and chosen location
        chosen_location = options.sensing_locations[chosen_location_index]
        drone_location = options.drones_locations[drone]
        distance = np.linalg.norm(chosen_location - drone_location)

        distance_multiplier = 0.03

        # Normalizing distance
        # between 0 and 1. The smaller the distance, the higher the reward.
        normalized_location_distance = (
            1 - (1 / (1 + np.exp(-distance * distance_multiplier)))
        ) * 2.0

        aoi_weight = 0.7
        distance_weight = 0.3
        reward = (
            normalized_diff_aoi_component * aoi_weight
            + normalized_location_distance * distance_weight
            + np.random.random() * (1.0 - aoi_weight - distance_weight)
        )

        if reward >= highest_reward:
            best_action = chosen_location_index
            highest_reward = reward

    return best_action


options = Namespace(
    cycle_length=1.0,
    sensing_locations=np.array([]),
    aois=np.array([]),
    sensing_data_amounts=np.array([]),
    drones_locations=np.array([]),
    sensing_locations_amount=10,
    cycle_stages=np.array([]),
    data_transmission_cycle=0.0,
    current_cycle=[0],
    grid_size=100.0,
    drones_amount=5,
    drone_max_speed=5.0,
    drone_bandwidth=0.5,
    total_location_data=1.5,
    cycles_num=300,
    aois_vec=np.array([[]]),
    drones_vec=np.array([[]]),
    chosen_loc_vec=np.array([[]]),
    cycle_stages_vec=np.array([[]]),
    random=False,
)


def main(
    grid_size=options.grid_size,
    sensing_locations_amount=options.sensing_locations_amount,
    drones_amount=options.drones_amount,
    drone_max_speed=options.drone_max_speed,
    drone_bandwidth=options.drone_bandwidth,
    total_location_data=options.total_location_data,
    cycles_num=options.cycles_num,
    random=options.random,
):
    sensing_locations = (np.random.rand(sensing_locations_amount, 2) * grid_size) - (
        grid_size / 2
    )
    options.sensing_locations = sensing_locations
    options.sensing_locations_amount = sensing_locations_amount
    options.grid_size = grid_size
    options.drones_amount = drones_amount
    options.drone_max_speed = drone_max_speed
    options.drone_bandwidth = drone_bandwidth
    options.cycles_num = cycles_num
    options.random = random

    # -- BEGIN of the jupyter notebook's code (please refer to it for a more readable version)
    # Link to the online notebook: https://github.com/AlessioLuciani/distributed-uav-rl-protocol/blob/main/simulation.ipynb

    options.drones_locations = np.zeros(
        (options.drones_amount, 2), dtype=float)
    options.sensing_data_amounts = np.zeros(options.drones_amount, dtype=float)
    options.aois = np.zeros(options.sensing_locations_amount, dtype=int)
    options.chosen_locations = np.zeros(options.drones_amount, dtype=int)
    options.cycle_stages = np.zeros(options.drones_amount, dtype=int)
    options.data_transmission_cycle = options.drone_bandwidth * options.cycle_length

    options.aois_vec = np.zeros(
        (cycles_num, options.sensing_locations_amount), dtype=int
    )
    options.drones_vec = np.zeros(
        (cycles_num, options.drones_amount, 2), dtype=float)
    options.chosen_loc_vec = np.zeros(
        ((cycles_num, options.drones_amount)), dtype=int)
    options.cycle_stages_vec = np.zeros(
        (cycles_num, options.drones_amount), dtype=int)

    class DurpEnv(py_environment.PyEnvironment):
        def __init__(self, drone):
            self._drone = drone
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.int32,
                minimum=0,
                maximum=options.sensing_locations_amount - 1,
                name="action",
            )
            self._observation_spec = array_spec.BoundedArraySpec(
                shape=(2,),
                minimum=(-options.grid_size / 2),
                maximum=(options.grid_size / 2),
                dtype=np.float64,
            )

        def action_spec(self):
            return self._action_spec

        def observation_spec(self):
            return self._observation_spec

        def _reset(self):
            return ts.restart(np.array([0.0, 0.0], dtype=np.float64))

        def _step(self, action):
            chosen_location_index = int(action)
            accumulated_aoi = get_accumulated_aoi(options.current_cycle[0])
            options.aois[chosen_location_index] = options.current_cycle[0]
            new_accumulated_aoi = get_accumulated_aoi(options.current_cycle[0])
            aoi_multiplier = 0.05
            normalized_diff_aoi_component = (
                (
                    1
                    / (
                        1
                        + np.exp(
                            -(accumulated_aoi - new_accumulated_aoi) *
                            aoi_multiplier
                        )
                    )
                )
                - 0.5
            ) * 2.0
            chosen_location = options.sensing_locations[chosen_location_index]
            drone_location = options.drones_locations[self._drone]
            distance = np.linalg.norm(chosen_location - drone_location)
            distance_multiplier = 0.03
            normalized_location_distance = (
                1 - (1 / (1 + np.exp(-distance * distance_multiplier)))
            ) * 2.0
            aoi_weight = 0.7
            distance_weight = 0.3
            reward = (
                normalized_diff_aoi_component * aoi_weight
                + normalized_location_distance * distance_weight
                + np.random.random() * (1.0 - aoi_weight - distance_weight)
            )

            return ts.transition(chosen_location, reward=reward)

    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    environments = []
    agents = []
    for drone in range(options.drones_amount):
        durp_env = DurpEnv(drone)
        train_env = tf_py_environment.TFPyEnvironment(durp_env)
        q_net = q_network.QNetwork(
            train_env.observation_spec(), train_env.action_spec()
        )

        train_step_counter = tf.Variable(0)

        agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            train_step_counter=train_step_counter,
        )

        agent.initialize()
        agents.append(agent)
        environments.append(train_env)

    num_iterations = 5
    intermediate_iterations = 5
    eval_interval = 10
    initial_collect_steps = 1
    collect_steps_per_iteration = 1
    batch_size = 64

    def compute_avg_return(environment, policy, num_episodes=10):
        total_return = 0.0
        for _ in range(num_episodes):
            time_step = environment.reset()
            episode_return = 0.0

            for m in range(10):
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def collect_step(environment, policy, buffer, drone):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        options.drones_locations[drone] = next_time_step.observation
        traj = trajectory.from_transition(
            time_step, action_step, next_time_step)

        buffer.add_batch(traj)

    def collect_data(env, policy, buffer, steps, drone):
        for step in range(1, steps + 1):
            options.current_cycle[0] = step
            collect_step(env, policy, buffer, drone)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agents[0].collect_data_spec, batch_size=environments[0].batch_size
    )

    random_policy = random_tf_policy.RandomTFPolicy(
        environments[0].time_step_spec(), environments[0].action_spec()
    )

    reset_aois()
    reset_drones_locations()

    collect_data(
        environments[0], random_policy, replay_buffer, initial_collect_steps, 0
    )

    reset_aois()
    reset_drones_locations()

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
    ).prefetch(3)
    iterator = iter(dataset)

    # Reset the train step
    returns = np.zeros(
        (options.drones_amount, (num_iterations // eval_interval) + 1), dtype=np.float64
    )
    for k in range(len(agents)):
        avg_return = compute_avg_return(environments[k], agents[k].policy)
        returns[k][0] = avg_return

    for i in range(num_iterations):
        reset_aois()
        reset_drones_locations()
        if i % 100 == 0:
            print("----------------", i)

        for j in range(intermediate_iterations):
            for k in range(len(agents)):
                agent = agents[k]
                env = environments[k]

                # Collect a few steps using collect_policy and save to the replay buffer.
                collect_data(
                    env,
                    agent.collect_policy,
                    replay_buffer,
                    collect_steps_per_iteration,
                    k,
                )

        for k in range(len(agents)):
            agent = agents[k]

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss
            print("Loss:", train_loss)

        if i % eval_interval == 0:
            for k in range(len(agents)):
                agent = agents[k]
                avg_return = compute_avg_return(environments[k], agent.policy)
                returns[k][(num_iterations // eval_interval)] = avg_return

    time_steps = []
    for drone in range(options.drones_amount):
        time_steps.append(environments[drone].reset())

    reset_aois()
    reset_drones_locations()

    # -- END of the jupyter notebook's code
    # -- Start of the simulation

    for cycle in range(1, options.cycles_num + 1):
        if cycle % 10 == 1:
            print(get_accumulated_aoi(cycle))
            print(options.aois)
            print("----------------")
        for drone in range(options.drones_amount):
            if options.cycle_stages[drone] == 0:
                agent = agents[drone]
                env = environments[drone]
                chosen_location_index = -1
                if random:
                    chosen_location_index = randrange(
                        options.sensing_locations_amount)
                    options.aois[chosen_location_index] = cycle
                else:
                    options.current_cycle[0] = cycle
                    policy_step = agent.policy.action(time_steps[drone]).replace(
                        action=tf.constant(
                            [get_best_action(drone)], dtype=np.int32)
                    )
                    new_step = env.step(policy_step.action)
                    time_steps[drone] = new_step
                    chosen_location_index = int(policy_step.action)
                options.chosen_locations[drone] = chosen_location_index
                options.cycle_stages[drone] = 1
            elif (
                options.drones_locations[drone]
                != options.sensing_locations[options.chosen_locations[drone]]
            ).all():
                traj = get_trajectory(drone)
                new_location = options.drones_locations[drone] + traj
                options.drones_locations[drone] = new_location
            elif options.sensing_data_amounts[drone] == 0.0:
                options.cycle_stages[drone] = 2
                options.sensing_data_amounts[drone] = options.total_location_data
                options.cycle_stages[drone] = 3
            else:
                options.sensing_data_amounts[drone] = np.max(
                    [
                        options.sensing_data_amounts[drone]
                        - options.data_transmission_cycle,
                        0.0,
                    ]
                )
                if options.sensing_data_amounts[drone] == 0.0:
                    options.cycle_stages[drone] = 0
        options.cycle_stages_vec[cycle - 1] = options.cycle_stages
        options.aois_vec[cycle - 1] = [
            get_location_aoi(cycle, index)
            for index in range(options.sensing_locations_amount)
        ]
        options.drones_vec[cycle - 1] = options.drones_locations
        options.chosen_loc_vec[cycle - 1] = options.chosen_locations

    app = QApplication(sys.argv)
    screen_w, screen_h = get_screen_resolution(app)
    simulator = Simulator(screen_w, screen_h)
    simulator.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    Fire(main)
