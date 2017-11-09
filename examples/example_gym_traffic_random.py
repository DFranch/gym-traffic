import gym
import gym_traffic
from gym.wrappers import Monitor
import gym
import time
env = gym.make('Traffic-Simple-gui-v0')
monitor = False
#env = gym.make('Traffic-Simple-cli-v0')
if monitor:
    env = Monitor(env, "output/traffic/simple/random", force=True)
for i_episode in range(500):
    observation = env.reset()
    for t in range(1000):
        env.render()
        print(observation)
        action = env.action_space.sample()
        print(env.action_space)
        time.sleep(0.1)
        print(action)
        observation, reward, done, info = env.step(action)
        print("Reward: {}".format(reward))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
