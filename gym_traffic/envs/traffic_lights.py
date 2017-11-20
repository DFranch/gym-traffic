class TrafficLight(object):
    def __init__(self, id, actions):
        self.state = 0
        self.step = 0
        self.id = id
        self.actions = actions

    def signal(self):

        return self.actions[self.state]

    def act(self, action):
        if action != self.state and self.action_allowed(action):
            self.state = action
            self.step = 0
        else:
            self.step += 1
        return self.signal()

    def action_allowed(self, action):
        return True


class TrafficLightTwoWay(TrafficLight):
    def __init__(self, id, yield_time=5):
        super(TrafficLightTwoWay, self).__init__(id=id, actions=["ryry", "GrGr", "rGrG", "yryr"])
        self.yield_time = yield_time

    def action_allowed(self, action):
        return True