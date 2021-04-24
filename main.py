from icecream import ic
import gym
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

import time
import random

# Constants
GAMMA = 0.99
# LIMIT = 1e-5
LIMIT = 0.3
NO_OF_ITERS = 200000
POL_LIMIT = 1e-10
POL_LIMIT = 0.3
FROZEN_LAKE = 'FrozenLake-v0'
BLACKJACK = 'Blackjack-v0'

### Value Iteration
SR_TRANS_PROB = 0
SR_NEXT_STATE = 1
SR_REW_PROB = 2

### Q-learning
LEARN_RATE = 0.1
EPISODE_COUNT = 20000
MAX_STEPS_FOR_EPISODE = 200
EXPLORE_DECAY = 0.001
TOP_EXPLORE_RATE = 1
BOTTOM_EXPLORE_RATE = 0.01
EXPLORE_RATE = 1

############# VALUE ITERATION #####################
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
    total = 0

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
    print(f'Threshold = {total}')
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

############# POLICY ITERATION #####################
def find_val_func(environ, pol, gamma=1.0):
    val_table = np.zeros(environ.nS)
    while True:
        new_val_table = np.copy(val_table)
        for state in range(environ.nS):
            action = pol[state]
            val_table[state] = sum(transition_prob * (rew_prob + gamma * new_val_table[new_state])
                                   for transition_prob, new_state, rew_prob, info \
                                   in environ.P[state][action])
        f = np.fabs(new_val_table - val_table)
        if np.sum(f) <= POL_LIMIT:
            break
    return val_table

def find_pol(environ, val_table, gamma = 1.0):
    # environ = gym.make(environ_name,
    #                    desc=map,
    #                    is_slippery=True)
    no_of_states = environ.observation_space.n
    no_of_actions = environ.action_space.n
    pol = np.zeros(no_of_states)
    # Looping through all of the states...
    for state in range(no_of_states):
        #...and all of the actions in that state...
        qtable = np.zeros(no_of_actions)
        for action in range(no_of_actions):
            # Looking though all of the probabilities...
            for state_reward in environ.P[state][action]:
                # ... and fill the Q-table.
                transition_prob, new_state, rew_prob, info = state_reward
                qtable[action] += (transition_prob * (rew_prob + \
                                                      gamma * val_table[new_state]))
        # Create policy by grabbing all of the best moves in the Q-table
        pol[state] = np.argmax(qtable)
    return pol

def pol_iter(map, environ_name, gamma = 1.0, iters = NO_OF_ITERS):
    environ = gym.make(environ_name,
                       desc=map,
                       is_slippery=True)
    no_of_states = environ.observation_space.n
    no_of_actions = environ.action_space.n
    rand_pol = np.zeros(no_of_states)

    for iter in range(iters):
        new_val_func = find_val_func(environ, rand_pol, gamma=gamma)
        new_pol = find_pol(environ, new_val_func, gamma)
        if(np.all(rand_pol == new_pol)):
            ic(f'Converged at step {iter+1}')
            break
        rand_pol = new_pol
    return new_pol


#############  Q-LEARNING #####################
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

    return q_table

def out_map(map, pols, alg_name):
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
    pol_as_arrows = ''.join(pol_as_arrows)

    # Print out the policy
    print(f'Final value {alg_name} for Frozen Lake')
    print('--------------------------------------------')
    joined_map = ''.join(map)
    col_cnt = 0
    for i in range(len(pol_as_arrows)):
        if joined_map[i] == 'S' or \
                joined_map[i] == 'G' or \
                joined_map[i] == 'H':
            print(joined_map[i], end='')
        else:
            print(pol_as_arrows[i], end='')
        col_cnt += 1
        if col_cnt % 15 == 0:
            print('')


if __name__ == '__main__':

    # Do value iteration
    rand_map_16 = generate_random_map(size=15, p=0.7)
    # pol = find_best_policy(rand_map_16, GAMMA, val_iter(rand_map_16, GAMMA, FROZEN_LAKE), FROZEN_LAKE)
    # out_map(rand_map_16, pol, 'Value Iteration')

    # Do policy iteration
    # pol = pol_iter(rand_map_16, FROZEN_LAKE, GAMMA, NO_OF_ITERS)
    # out_map(rand_map_16, pol, 'Policy Iteration')


    ####### CHARTS FOR VALUE ITERATION ########
    # Iterations vs Time plot for value iteration
    times = []
    iters = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 1000000]
    for iter in iters:
        tic = time.perf_counter()
        # Generate a 10x10 map with 0.7 probility tile is slippery.
        pols = val_iter(rand_map_16, GAMMA, FROZEN_LAKE, iters=iter)
        opt_pol = find_best_policy(rand_map_16,
                                   GAMMA,
                                   pols,
                                   FROZEN_LAKE)
        # ic(opt_pol)
        toc = time.perf_counter()
        print(f'Took {toc - tic} seconds')
        times.append(toc - tic)

    df_for_vl = pd.DataFrame(data={'Iterations': iters,
                                   'Time(seconds)': times})
    ic(df_for_vl.dtypes)
    # Do the time plot
    sns.lineplot(x='Iterations',
                 y='Time(seconds)',
                 data=df_for_vl,
                 palette='pastel')
    sns.set_style('dark')

    plt.title(f'Time vs Iterations for Frozen Lake Value Iteration', fontsize=13)
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('outputs/fl_vi_iter_time.png')


    # Gammas vs Time plot for value iteration
    times = []
    # iters = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 1000000]
    gammas = np.arange(0.05, 1.0, 0.05)
    for i in gammas:
        tic = time.perf_counter()
        pols = val_iter(rand_map_16, i, FROZEN_LAKE, iters=NO_OF_ITERS)
        opt_pol = find_best_policy(rand_map_16,
                                   i,
                                   pols,
                                   FROZEN_LAKE)
        # ic(opt_pol)
        toc = time.perf_counter()
        print(f'Took {toc - tic} seconds')
        times.append(toc - tic)

    df_for_vl = pd.DataFrame(data={'Discounts': gammas,
                                   'Time(seconds)': times})
    ic(df_for_vl.dtypes)
    # Do the time plot
    sns.lineplot(x='Discounts',
                 y='Time(seconds)',
                 data=df_for_vl,
                 palette='pastel')
    sns.set_style('dark')

    plt.title(f'Time vs Discounts for Frozen Lake Value Iteration', fontsize=13)
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('outputs/fl_vi_gamma_time.png')


    ####### CHARTS FOR POLICY ITERATION ########
    # Iterations vs Time plot for policy iteration
    times = []
    iters = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 1000000]
    for iter in iters:
        tic = time.perf_counter()
        pol = pol_iter(rand_map_16, FROZEN_LAKE, GAMMA, iters=iter)
        # ic(opt_pol)
        toc = time.perf_counter()
        print(f'Took {toc - tic} seconds')
        times.append(toc - tic)

    df_for_vl = pd.DataFrame(data={'Iterations': iters,
                                   'Time(seconds)': times})
    ic(df_for_vl.dtypes)
    # Do the time plot
    sns.lineplot(x='Iterations',
                 y='Time(seconds)',
                 data=df_for_vl,
                 palette='pastel')
    sns.set_style('dark')

    plt.title(f'Time vs Iterations for Frozen Lake Policy Iteration', fontsize=13)
    # plt.legend(loc='upper right')
    plt.show()
    plt.savefig('outputs/fl_pi_iters_time.png')


    # Gammas vs Time plot for policy iteration
    times = []
    # iters = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 1000000]
    gammas = np.arange(0.05, 1.0, 0.05)
    for i in gammas:
        tic = time.perf_counter()
        pol = pol_iter(rand_map_16, FROZEN_LAKE, i, NO_OF_ITERS)
        # ic(opt_pol)
        toc = time.perf_counter()
        print(f'Took {toc - tic} seconds')
        times.append(toc - tic)

    df_for_vl = pd.DataFrame(data={'Discounts': gammas,
                                   'Time(seconds)': times})
    ic(df_for_vl.dtypes)
    # Do the time plot
    sns.lineplot(x='Discounts',
                 y='Time(seconds)',
                 data=df_for_vl,
                 palette='pastel')
    sns.set_style('dark')

    plt.title(f'Time vs Discounts for Frozen Lake Policy Iteration', fontsize=13)
    # plt.legend(loc='upper right')
    plt.show()
    plt.savefig('outputs/fl_pi_gamma_time.png')


    # writer = SummaryWriter(comment== 'vs. iteration')
    # top_rew = 0.0
    # iter = 0
    # env_choice, alg_choice = '', ''
    # # Ask user which environment/algorithm
    # env_choice = input('1. Frozen Lake, 2. nChain or 3. Run Everything')
    # if env_choice != '3':
    #     alg_choice = input('1. Value, 2. Policy or 3. Q-learning')
    #
    # tic = time.perf_counter()
    # if env_choice == '1':
    #     if alg_choice == '1':
    #
    #         opt_pol = find_best_policy(GAMMA,
    #                                    val_iter(GAMMA, FROZEN_LAKE),
    #                                    FROZEN_LAKE)
    #         print(f'Optimum policy for frozen lake...\n\n{opt_pol}')
    #     elif alg_choice == '2':
    #         print('Not yet implemented')
    #     elif alg_choice == '3':
    #         print(f'Q-table\n\n{q_learning(GAMMA, LEARN_RATE, FROZEN_LAKE)}')
    #
    # elif env_choice == '2':
    #     if alg_choice == '3':
    #         print(f'Q-table\n\n{q_learning(GAMMA, LEARN_RATE, BLACKJACK)}')
    #
    # print(f'Took {time.perf_counter() - tic} seconds')

