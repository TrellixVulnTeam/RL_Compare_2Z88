import random

from icecream import ic
import gym
import numpy as np

import time

FROZEN_LAKE = 'FrozenLake-v0'
GAMMA = 0.99
LEARN_RATE = 0.1
EPISODE_COUNT = 20000
MAX_STEPS_FOR_EPISODE = 200
EXPLORE_DECAY = 0.001
TOP_EXPLORE_RATE = 1
BOTTOM_EXPLORE_RATE = 0.01
EXPLORE_RATE = 1

def q_learning(gamma, learn_rate, environ_name):
    """Apply the Q-learning algorithm to the given Open AI Gym environment.

    :param gamma: Discount factor from 0 to 1
    :param learn_rate:
    :param environ_name: Name used in Open AI gym
    :return: The q table created.
    """

    environ = gym.make(environ_name)
    no_of_states = environ.observation_space.n
    no_of_actions = environ.action_space.n
    q_table = np.zeros((no_of_states, no_of_actions))
    total_rewards = []
    print(f'Starting Q-Learning for {environ_name}')

    ## Run through the speicified number of episodes...
    for episode in range(EPISODE_COUNT):
        complete = False
        state = environ.reset()
        curr_episode_rew = 0

        # ... and for each episode, run through the given number of steps...
        for step in range(MAX_STEPS_FOR_EPISODE):
            # ... first work through the exploitation/exploration trade-off...
            if random.uniform(0, 1) > EXPLORE_RATE:
                action = np.argmax((q_table[state, :]))
            else:
                action = environ.action_space.sample()

            # ... do the step itself...
            next_state, reward, complete, _ = environ.step(action)

            # ... then update the Q-table
            q_table[state, action] = q_table[state, action] * (1 - LEARN_RATE) + \
                                     LEARN_RATE * (reward + gamma *
                                     np.max(q_table[next_state, :]))
            curr_episode_rew += reward
            state = next_state
            if complete:
                break


    explore_rate = BOTTOM_EXPLORE_RATE + (TOP_EXPLORE_RATE - BOTTOM_EXPLORE_RATE) * \
                   np.exp(-EXPLORE_DECAY * episode)
    total_rewards.append(curr_episode_rew)

    # Calculate and print the average reward per thousand episodes
    # rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),
    #                                          num_episodes / 1000)
    # count = 1000
    # print("\tAverage reward per thousand episodes")
    # print("\t____________________________________\n")
    # for r in rewards_per_thousand_episodes:
    #     print(count, ": ", str(sum(r / 1000)))
#     count += 1000

    return q_table


if __name__ == '__main__':
    print(f'Q-table\n\n{q_learning(GAMMA, LEARN_RATE, FROZEN_LAKE)}')



#
# print("\n\nPress [ENTER] to watch a simulation of this training...")
# input()
# os.system('cls')
# clear_output(wait=True)
#
# # Simulation to see the agent after the training
# for episode in range(3):
#     state = env.reset()
#     done = False
#     print("\tEPISODE ", episode + 1)
#     print("\t_________\n")
#     time.sleep(1.5)
#
#     for step in range(max_steps_per_episode):
#         os.system('cls')
#         clear_output(wait=True)
#         env.render()
#         time.sleep(0.3)
#
#         action = np.argmax(q_table[state, :])
#         new_state, reward, done, info = env.step(action)
#
#         if done:
#             os.system('cls')
#             clear_output(wait=True)
#             env.render()
#             if reward == 1:
#                 print("\n--> You reached the goal! :)")
#                 time.sleep(3)
#             else:
#                 print("\n--> You fell through a hole! :(")
#                 time.sleep(3)
#             os.system('cls')
#             clear_output(wait=True)
#             break
#         state = new_state
#
# env.close()
# print("\n\nPress [ENTER] to watch a simulation of this training...")
# input()
# os.system('cls')
# clear_output(wait=True)
#
# # Simulation to see the agent after the training
# for episode in range(3):
#     state = env.reset()
#     done = False
#     print("\tEPISODE ", episode + 1)
#     print("\t_________\n")
#     time.sleep(1.5)
#
#     for step in range(max_steps_per_episode):
#         os.system('cls')
#         clear_output(wait=True)
#         env.render()
#         time.sleep(0.3)
#
#         action = np.argmax(q_table[state, :])
#         new_state, reward, done, info = env.step(action)
#
#         if done:
#             os.system('cls')
#             clear_output(wait=True)
#             env.render()
#             if reward == 1:
#                 print("\n--> You reached the goal! :)")
#                 time.sleep(3)
#             else:
#                 print("\n--> You fell through a hole! :(")
#                 time.sleep(3)
#             os.system('cls')
#             clear_output(wait=True)
#             break
#         state = new_state
#
# env.close()
