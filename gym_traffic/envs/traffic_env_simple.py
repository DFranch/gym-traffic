from .traffic_env import TrafficEnv
from .traffic_lights import TrafficLightTwoWay
import os


class TrafficEnvSimple(TrafficEnv):
    def __init__(self, mode="gui"):
        lights = [TrafficLightTwoWay(id="0", yield_time=3)]
        loops = ["loop{}".format(i) for i in range(12)]
        lanes = ["n_0_0", "s_0_0", "e_0_0", "w_0_0", "0_n_0", "0_s_0", "0_e_0", "0_w_0"]
        basepath = os.path.join(os.path.dirname(__file__), "config", "simple")
        netfile = os.path.join(basepath, "traffic.net.xml")
        routefile = os.path.join(basepath, "traffic.rou.xml")
        guifile = os.path.join(basepath, "view.settings.xml")
        addfile = os.path.join(basepath, "traffic.add.xml")
        exitloops = ["loop4", "loop5", "loop6", "loop7"]
        self.viewer = None
        super(TrafficEnvSimple, self).__init__(mode=mode, lights=lights, netfile=netfile, routefile=routefile,
                                               guifile=guifile, loops=loops, addfile=addfile, simulation_end=300,
                                               lanes=lanes, exitloops=exitloops)

    def route_sample(self):
        ns = 0.02
        ew = 0.08
        return {"ns": ns,
                "sn": ns,
                "ew": ew,
                "we": ew
                }
