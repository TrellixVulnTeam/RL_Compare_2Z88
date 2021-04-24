from icecream import ic
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

import time

# Constants
GAMMA = 1.0
LIMIT = 1e-20
NO_OF_ITERS = 100000
FROZEN_LAKE = 'FrozenLake-v0'
SR_TRANS_PROB = 0
SR_NEXT_STATE = 1
SR_REW_PROB = 2

def val_iter(map, gamma, environ_name, iters=NO_OF_ITERS):
    """Value iteration

    :param map: The map to be used
    :param gamma: Discount factor from 0 to 1
    :param environ_name: Name used in Open AI gym
    :return: Latest value table
    """

    environ = gym.make(environ_name,
                       desc=map,
                       is_slippery=True)
    no_of_states = environ.observation_space.n
    no_of_actions = environ.action_space.n
    val_table = np.zeros(iters)

    print(f'Running value iteration for {environ_name}')
    ## Loop through the iterations set
    for iter in range(iters):
        if iter % 10000 == 0:
            ic(iter)
        new_val_table = np.copy(val_table)

        # Loop through each of the states...
        for state in range(no_of_states):
            q_val = []
            # ... and for each state, check all of the actions...
            for action in range(no_of_actions):
                rewards = []
                # ... and for each action find all the Q values.
                for state_reward in environ.P[state][action]:
                    # Calculate the Q value for the current transition
                    transition_prob, new_state, rew_prob, info = state_reward
                    rewards.append((transition_prob * (rew_prob + \
                                    gamma * new_val_table[new_state])))
                total_q_val = np.sum(rewards)
                q_val.append(total_q_val)

            # Find the best action for this state and add it to the value table
            val_table[state] = max(q_val)

        # Check to see if the code has found the convergence limit so we can
        # stop.
        absolutes = np.fabs(new_val_table - val_table)
        total = np.sum(absolutes)
        if total <= LIMIT:
            break
    print(f'Converged after {iters} iterations.')
    return val_table


def find_best_policy(map, gamma, val_table, environ_name):
    """Use the value table to find the best policy

    :param gamma: Discount factor from 0 to 1
    :param val_table: The table to search
    :return: The best policy found from the value table
    """
    # environ = gym.make(environ_name)
    environ = gym.make(environ_name,
                       desc=map,
                       is_slippery=True)
    no_of_states = environ.observation_space.n
    no_of_actions = environ.action_space.n
    best_policy = np.zeros(no_of_states)
    print(f'Finding best policy for {environ_name}')

    # Loop through all of the states...
    for state in range(no_of_states):
        q_table = np.zeros(no_of_actions)
        # ... and all of the actions...
        for action in range(no_of_actions):
            # ... and all the next state/rewards...
            for state_reward in environ.P[state][action]:
                # ... to build the Q table...
                q_table[action] += (state_reward[SR_TRANS_PROB] * \
                                    (state_reward[SR_REW_PROB] + gamma * \
                                   val_table[state_reward[SR_NEXT_STATE]]))
        # ... and finally use the Q table to find the best policy.
        best_policy[state] = np.argmax(q_table)
    return best_policy





if __name__ == '__main__':
    rand_map_16 = generate_random_map(size=15, p=0.7)
    pols = find_best_policy(rand_map_16, GAMMA, val_iter(rand_map_16, GAMMA, FROZEN_LAKE), FROZEN_LAKE)
    # Convert policy to arrows
    pol_as_arrows = []
    for pol in pols:
        if pol == 0:
            pol_as_arrows.append('<')
        elif pol == 1:
            pol_as_arrows.append('V')
        elif pol == 2:
            pol_as_arrows.append('>')
        elif pol == 3:
            pol_as_arrows.append('^')
    print('Final value iteration policy for Frozen Lake')
    print('--------------------------------------------')
    joined_map = ''.join(rand_map_16)
    col_cnt = 0
    for i in len(pol_as_arrows):
        if joined_map[i] == 'S' or \
           joined_map[i] == 'G' or \
           joined_map[i] == 'H':
            print(joined_map[i])
        else:
            print(pol_as_arrows[i], end='')
        col_cnt += 1
        if col_cnt % 15 == 0:
            print('')


    # times = []
    # iters = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    # for iter in iters:
    #     tic = time.perf_counter()
        # Generate a 10x10 map with 0.7 probility tile is slippery.
        # pols = val_iter(rand_map_15, GAMMA, FROZEN_LAKE, iters=iter)
        # opt_pol = find_best_policy(rand_map_15,
        #                            GAMMA,
        #                            pols,
        #                            FROZEN_LAKE)
        # ic(opt_pol)
        # toc = time.perf_counter()
        # print(f'Took {toc - tic} seconds')
        # times.append(toc - tic)
    #
    # df_for_vl = pd.DataFrame(data={'Iterations': iters,
    #                                'Time(seconds)': times})
    # ic(df_for_vl.dtypes)
    # Do the time plot
    # sns.lineplot(x='Iterations',
    #              y='Time(seconds)',
    #              data=df_for_vl,
    #              palette='pastel')
    # sns.set_style('dark')
    #
    # plt.title(f'Time vs Iterations for Frozen Lake Value Iteration', fontsize=13)
    # plt.legend(loc='upper right')
    # plt.show()
    #
    # writer = SummaryWriter(comment== 'vs. iteration')
    # top_rew = 0.0
    # iter = 0
#


