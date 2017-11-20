import gym

from gym_traffic.agents import DQN
import time

def main():
    env = gym.make('Traffic-Simple-cli-v0')
    gamma = 0.9
    epsilon = .95

    trials = 1000
    trial_len = 500
    total_reward = 0
    elapsed_time = 0
    last_epsilon = 0
    counter = 0
    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    for trial in range(trials):
        cur_state = env.reset()
        start_time = time.time()
        counter += 1
        print("Run number: {}".format(counter))
        print("Time elapsed: {}".format(elapsed_time))
        print("Mean Reward: {}".format(total_reward / trial_len))
        print("Last Epsilon: {}".format(last_epsilon))
        total_reward = 0
        elapsed_time = 0
        last_epsilon = 0

        for step in range(trial_len):
            action, epsilon = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)
            #print("Reward: {}".format(reward))

            # reward = reward if not done else -20
            # new_state = new_state.reshape(1, 2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.target_train()  # iterates target model


            total_reward += reward
            cur_state = new_state
            elapsed_time = time.time() - start_time
            last_epsilon = epsilon

            if done:
                break
        # if step >= 199:
        #     print("Failed to complete in trial {}".format(trial))
        #     if step % 10 == 0:
        #         dqn_agent.save_model("trial-{}.model".format(trial))
        # else:
        #     print("Completed in {} trials".format(trial))
        #     dqn_agent.save_model("success.model")
        #     break


if __name__ == "__main__":
    main()
