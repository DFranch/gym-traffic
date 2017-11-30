
import numpy as np
import random
import pandas as pd
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = .95
        self.epsilon = .95
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0005
        self.tau = .125

        self.observations_df = pd.DataFrame(columns=[
            "#run",
            "waiting_time",
            "action",
            "confidence"
        ])

        self.observations_file_name = 'observations_{0}_run.csv'.format(int(time.time()))

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(256, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.env.action_space.n, activation="softmax"))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        # print("Epsilon: {}".format(self.epsilon))
        if np.random.random() < self.epsilon:
            sampel = self.env.action_space.sample()
            #print("Taking random action: {0}".format(sampel))
            return sampel, self.epsilon, "RANDOM"
        else:
            #print("self.model.predict(state)[0]): {}".format(self.model.predict(state)[0]))
            action = np.argmax(self.model.predict(state)[0])
            confidence = max(self.model.predict(state)[0])
            # print("Taking predicted action: {0}, with confidence: {1}".format(action, confidence))
            return action, self.epsilon, confidence


    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 64
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                #print(self.target_model.predict(new_state)[0])
                #print("Q_future: {}".format(Q_future))
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

    def add_observations_to_df(self, observations):
        self.observations_df = self.observations_df.append(
            pd.Series(observations, index=self.observations_df.columns),
            ignore_index=True
        )

    def add_observations_to_csv(self):
        self.observations_df.to_csv(self.observations_file_name, index=False, header=False, mode="a")
        self.observations_df = pd.DataFrame(columns=[
            "#run",
            "waiting_time",
            "action",
            "confidence"
        ])
