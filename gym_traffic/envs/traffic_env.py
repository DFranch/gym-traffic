from gym import Env
from gym import error, spaces, utils
from gym.utils import seeding
from scipy.misc import imread
from gym import spaces
from .multi_discrete import DiscreteToMultiDiscrete
from string import Template
import os, sys
import numpy as np
import math
import time

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import traci.constants as tc


class TrafficEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, lights, netfile, routefile, guifile, addfile, loops=[], lanes=[], exitloops=[],
                 tmpfile="tmp.rou.xml", mode="gui", detector="detector0", simulation_end=3600, sleep_between_restart=1):
        # "--end", str(simulation_end),
        self.simulation_end = simulation_end
        self.sleep_between_restart = sleep_between_restart
        self.mode = mode
        self._seed()
        self.loops = loops
        self.exitloops = exitloops
        self.loop_variables = [tc.LAST_STEP_MEAN_SPEED, tc.LAST_STEP_TIME_SINCE_DETECTION, tc.LAST_STEP_VEHICLE_NUMBER]
        self.lanes = lanes
        self.detector = detector
        args = ["--net-file", netfile, "--route-files", tmpfile, "--additional-files", addfile]
        if mode == "gui":
            binary = "sumo-gui"
            args += ["-S", "-Q", "--gui-settings-file", guifile]
        else:
            binary = "sumo"
            args += ["--no-step-log"]

        with open(routefile) as f:
            self.route = f.read()
        self.tmpfile = tmpfile
        # self.pngfile = pngfile
        self.sumo_cmd = [binary] + args
        self.sumo_step = 0
        self.lights = lights
        print(self.lights)
        self.action_space = spaces.Discrete(4)
        # self.action_space = DiscreteToMultiDiscrete(
        #    spaces.MultiDiscrete([[0, len(light.actions) - 1] for light in self.lights]))

        trafficspace = spaces.Box(low=float('-inf'), high=float('inf'),
                                  shape=(len(self.loops) * len(self.loop_variables),))
        lightspaces = [spaces.Discrete(len(light.actions)) for light in self.lights]
        self.observation_space = spaces.Box(low=0, high=1000, shape=(1, 13))

        self.sumo_running = False
        self.viewer = None

    def relative_path(self, *paths):
        os.path.join(os.path.dirname(__file__), *paths)

    def write_routes(self):
        self.route_info = self.route_sample()
        with open(self.tmpfile, 'w') as f:
            f.write(Template(self.route).substitute(self.route_info))

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def start_sumo(self):
        if not self.sumo_running:
            self.write_routes()
            traci.start(self.sumo_cmd)

            for loopid in self.loops:
                traci.inductionloop.subscribe(loopid, self.loop_variables)
            self.sumo_step = 0
            self.sumo_running = True
            self.screenshot()

    def stop_sumo(self):
        if self.sumo_running:
            traci.close()
            self.sumo_running = False

    def _reward(self):
        lanes = traci.trafficlights.getControlledLanes("0")
        edges = []
        reward = 0
        avg_edge_values = np.zeros(4)
        for lane in lanes:
            edges.append(traci.lane.getEdgeID(lane))

        for e_id in edges:
            edge_values = [
                traci.edge.getWaitingTime(e_id),  # 0
                traci.edge.getCO2Emission(e_id),  # 1
                traci.edge.getFuelConsumption(e_id),  # 7
                traci.edge.getLastStepVehicleNumber(e_id),  # 11
            ]
            if edge_values[3] > 0:
                edge_values[2] /= edge_values[3]
                edge_values[1] /= edge_values[3]
                edge_values[0] /= edge_values[3]
            avg_edge_values = np.add(avg_edge_values, edge_values)

        lightState = traci.trafficlights.getRedYellowGreenState("0")
        waitingFactor = -avg_edge_values[0] / 100
        if waitingFactor == 0:
            waitingFactor += 1
        co2_factor = -avg_edge_values[1] / 3000
        fuel_factor = -avg_edge_values[2]
        green_factor = 7 * (lightState.count("g") + lightState.count("G")) / len(lanes)
        yellow_factor = -0.5 * lightState.count("y") / len(lanes)
        red_factor = -2 * lightState.count("r") / len(lanes)

        reward += waitingFactor + co2_factor + fuel_factor + green_factor + yellow_factor + red_factor
        return reward
        # reward = 0.0
        # for lane in self.lanes:
        #    reward -= traci.lane.getWaitingTime(lane)
        # return reward
        # speed = traci.multientryexit.getLastStepMeanSpeed(self.detector)
        # count = traci.multientryexit.getLastStepVehicleNumber(self.detector)
        # if count == 0:
        #    return speed
        # reward = speed / count
        # print("Speed: {}".format(traci.multientryexit.getLastStepMeanSpeed(self.detector)))
        # print("Count: {}".format(traci.multientryexit.getLastStepVehicleNumber(self.detector)))
        # reward = np.sqrt(speed)
        # print "Reward: {}".format(reward)
        # return speed

    # reward = 0.0
    # for loop in self.exitloops:
    #    reward += traci.inductionloop.getLastStepVehicleNumber(loop)

    # return max(reward, 0)

    def _step(self, action):
        # action = self.action_space(action)
        self.start_sumo()
        self.sumo_step += 1
        # assert (len(action) == len(self.lights))
        for light in self.lights:
            signal = light.act(action)
            traci.trafficlights.setRedYellowGreenState(light.id, signal)
        traci.simulationStep()
        observation = self._observation()
        reward = self._reward()
        done = self.sumo_step > self.simulation_end
        self.screenshot()
        if done:
            self.stop_sumo()
        return observation, reward, done, self.route_info

    def screenshot(self):
        return ""
        # if self.mode == "gui":
        # traci.gui.screenshot("View #0", self.pngfile)

    def _observation(self):
        lanes = traci.trafficlights.getControlledLanes("0")
        edges = []
        avg_edge_values = np.zeros(13)
        for lane in lanes:
            edges.append(traci.lane.getEdgeID(lane))

        for e_id in edges:
            # print(traci.edge.getCO2Emission(e_id))
            edge_values = [
                traci.edge.getWaitingTime(e_id),  # 0
                traci.edge.getCO2Emission(e_id),  # 1
                traci.edge.getCOEmission(e_id),
                traci.edge.getHCEmission(e_id),
                traci.edge.getPMxEmission(e_id),
                traci.edge.getNOxEmission(e_id),
                traci.edge.getFuelConsumption(e_id),
                traci.edge.getLastStepMeanSpeed(e_id),  # 7
                traci.edge.getLastStepOccupancy(e_id),
                traci.edge.getLastStepLength(e_id),
                traci.edge.getTraveltime(e_id),
                traci.edge.getLastStepVehicleNumber(e_id),  # 11
                traci.edge.getLastStepHaltingNumber(e_id)
            ]
            if edge_values[11] > 0:
                edge_values[7] /= edge_values[11]
                edge_values[1] /= edge_values[11]
                edge_values[0] /= edge_values[11]
            avg_edge_values = np.add(avg_edge_values, edge_values)

        avg_edge_values /= len(edges)
        return avg_edge_values

    def _reset(self):
        self.stop_sumo()
        # sleep required on some systems
        if self.sleep_between_restart > 0:
            time.sleep(self.sleep_between_restart)
        self.start_sumo()
        observation = self._observation()
        return observation

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if self.mode == "gui":
            # img = imread(self.pngfile, mode="RGB")
            if mode == 'rgb_array':
                return ""
            elif mode == 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    self.viewer = rendering.SimpleImageViewer()
                    # self.viewer.imshow(img)
        else:
            raise NotImplementedError("Only rendering in GUI mode is supported. Please use Traffic-...-gui-v0.")
