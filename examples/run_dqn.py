import gym

from gym_traffic.agents import DQN


def main():
    env = gym.make('Traffic-Simple-gui-v0')
    gamma = 0.9
    epsilon = .95

    trials = 1000
    trial_len = 500

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    for trial in range(trials):
        cur_state = env.reset()
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)
            print("Reward: {}".format(reward))

            # reward = reward if not done else -20
            # new_state = new_state.reshape(1, 2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.target_train()  # iterates target model

            cur_state = new_state
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
