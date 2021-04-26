from icecream import ic
import gym
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import time
import random
import pickle

# Constants
GAMMA = 0.99
LIMIT = 1e-10
# LIMIT = 0.3
NO_OF_ITERS = 200000
POL_LIMIT = 1e-10
POL_LIMIT = 0.3
CHECK_COUNT = 1000
STEPS = 10000
FROZEN_LAKE = 'FrozenLake-v0'

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

GOAL_FOUND = 1
GOAL_NOT_FOUND = -1
MAX_DECAY_COMPARE = 0.01

############# VALUE ITERATION #####################
def val_iter(the_map, gamma, environ_name, iters=NO_OF_ITERS):
    """Value iteration

    :param map: The map to be used
    :param gamma: Discount factor from 0 to 1
    :param environ_name: Name used in Open AI gym
    :return: Latest value table
    """

    environ = gym.make(environ_name,
                       desc=the_map,
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


def find_best_policy(the_map, gamma, val_table, environ_name):
    """Use the value table to find the best policy

    :param gamma: Discount factor from 0 to 1
    :param val_table: The table to search
    :return: The best policy found from the value table
    """
    # environ = gym.make(environ_name)
    environ = gym.make(environ_name,
                       desc=the_map,
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

def pol_iter(the_map, environ_name, gamma = 1.0, iters = NO_OF_ITERS):
    environ = gym.make(environ_name,
                       desc=the_map,
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

def check_policy(the_map, environ_name, pol):
    environ=gym.make(environ_name,
                     desc=the_map,
                     is_slippery=True)
    win_cnt = 0
    for i in range(CHECK_COUNT):
        st = environ.reset()
        for t in range(STEPS):
            st, rew, finished, _ = environ.step(pol[st])
            if finished:
                if rew:
                    win_cnt = win_cnt + 1
                break
    print(f'Succeeded {win_cnt} times in a thousand')
    environ.close()
    return win_cnt


def build_map(size=15, prob=0.5):
    rand_map_16 = generate_random_map(size=size, p=prob)
    with open('map.txt', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(rand_map_16, filehandle)

def q_learning(environ_name, the_map,
               epsilon=1.0,
               epsilon_decay=0.99,
               alpha_decay=0.99,
               alpha=0.99,
               gamma=0.9):
    """Apply the Q-learning algorithm to the given Open AI Gym environment.

    :param gamma: Discount factor from 0 to 1
    :param learn_rate:
    :param environ_name: Name used in Open AI gym
    :return: The q table created.
    """
    print('Starting Q-learning for Frozen Lake')
    environ = gym.make(environ_name,
                       desc=the_map,
                       is_slippery=True)
    no_of_states = environ.observation_space.n
    no_of_actions = environ.action_space.n

    games = []

    q_table = np.zeros((no_of_states, no_of_actions))
    rews = []
    iterations = []
    opt_pol = [0] * no_of_states
    episodes = 10000
    environ = environ.unwrapped

    Q_vals = []
    for episode in range(episodes):
        state = environ.reset()
        t_rew = 0
        done = False

        Q_vals.append(np.sum(q_table[0, :]))
        max_steps = 1000000
        for k in range(max_steps):
            if done == True:
                break
            curr = state
            if np.random.rand() < (epsilon):
                action = np.argmax(q_table[curr, :])
            else:
                action = environ.action_space.sample()

            new_state, rew, complete, _ = environ.step(action)
            t_rew = t_rew + rew
            q_table[curr, action] += alpha * (
                    rew + gamma * np.max(q_table[state, :]) - q_table[curr, action])
        # Do the decaying
        alpha = max(MAX_DECAY_COMPARE, alpha * alpha_decay )
        epsilon = max(MAX_DECAY_COMPARE, epsilon * epsilon_decay)
        rews.append(t_rew)
        iterations.append(k)

    for k in range(no_of_states):
        opt_pol[k] = np.argmax(q_table[k, :])

    games.append(run_eps(environ, 0, opt_pol))

    Q_vals = np.array(Q_vals)
    Q_vals /= 1000

    print(f'optimal policy = {opt_pol}')

    return opt_pol, np.mean(games)

def run_eps(environ, pol, no_games=1000):
    total_reward = 0
    state = environ.reset()
    for i in range(no_games):
        complete = False
        while not complete:
            new_state, rew, complete, info = environ.step(pol[state])
            state = new_state
            total_reward= total_reward + total_reward
            if complete == True:
                state = environ.reset()
    return total_reward


if __name__ == '__main__':
    with open('map.txt', 'rb') as filehandle:
        rand_map_16 = pickle.load(filehandle)
    print('Map opened')
    # pol, rew = q_learning(FROZEN_LAKE, rand_map_16)
    # print(f'Final rewards = {rew}')
    # out_map(rand_map_16, pol, 'Q-learning')
    # 
    # pol_val = find_best_policy(rand_map_16, GAMMA, val_iter(rand_map_16, GAMMA, FROZEN_LAKE), FROZEN_LAKE)
    # out_map(rand_map_16, pol_val, 'Value Iteration')
    # Do policy iteration
    # pol_pol = pol_iter(rand_map_16, FROZEN_LAKE, GAMMA, NO_OF_ITERS)
    # out_map(rand_map_16, pol_pol, 'Policy Iteration')

    ####### CHARTS FOR VALUE ITERATION ########
    # Iterations vs Time plot for value iteration
    times = []
    win_cnt = []
    iters = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 1000000]
    for iter in iters:
        tic = time.perf_counter()
        pols = val_iter(rand_map_16, GAMMA, FROZEN_LAKE, iters=iter)
        opt_pol = find_best_policy(rand_map_16,
                                   GAMMA,
                                   pols,
                                   FROZEN_LAKE)
        # ic(opt_pol)
        toc = time.perf_counter()
        print(f'Took {toc - tic} seconds')
        times.append(toc - tic)
        win_cnt.append(check_policy(rand_map_16, FROZEN_LAKE, opt_pol))

    df_for_vl = pd.DataFrame(data={'Iterations': iters,
                                   'Wins per thousand runs': win_cnt,
                                   'Time(seconds)': times})
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

    sns.lineplot(y='Wins per thousand runs',
                 x='Iterations',
                 data=df_for_vl,
                 palette='pastel')
    sns.set_style('dark')

    plt.title(f'Wins vs Iterations for Frozen Lake Value Iteration', fontsize=13)
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('outputs/fl_vi_iter_wins.png')

    # Gammas vs Time plot for value iteration
    times = []
    win_cnt = []
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
        win_cnt.append(check_policy(rand_map_16, FROZEN_LAKE, opt_pol))

    df_for_vl = pd.DataFrame(data={'Discounts': gammas,
                                   'Wins per thousand runs': win_cnt,
                                   'Time(seconds)': times})
    # ic(df_for_vl.dtypes)
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

    # Do the wins plot
    # Rew vs discount
    sns.lineplot(x='Wins per thousand runs',
                 y='Discounts',
                 data=df_for_vl,
                 palette='pastel')
    sns.set_style('dark')

    plt.title(f'Discounts vs Wins for Frozen Lake Value Iteration', fontsize=13)
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('outputs/fl_vi_wins_discount.png')

    ####### CHARTS FOR POLICY ITERATION ########
    # Iterations vs Time plot for policy iteration
    times = []
    win_cnt = []
    iters = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 1000000]
    for iter in iters:
        tic = time.perf_counter()
        pol = pol_iter(rand_map_16, FROZEN_LAKE, GAMMA, iters=iter)
        # ic(opt_pol)
        toc = time.perf_counter()
        print(f'Took {toc - tic} seconds')
        times.append(toc - tic)
        win_cnt.append(check_policy(rand_map_16, FROZEN_LAKE, opt_pol))

    df_for_vl = pd.DataFrame(data={'Iterations': iters,
                                   'Wins per thousand runs': win_cnt,
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

    sns.lineplot(y='Wins per thousand runs',
                 x='Iterations',
                 data=df_for_vl,
                 palette='pastel')
    sns.set_style('dark')

    plt.title(f'Wins vs Iterations for Frozen Lake Policy Iteration', fontsize=13)
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('outputs/fl_pi_iter_wins.png')

    # Gammas vs Time plot for policy iteration
    times = []
    win_cnt = []
    # iters = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 1000000]
    gammas = np.arange(0.05, 1.0, 0.05)
    for i in gammas:
        tic = time.perf_counter()
        pol = pol_iter(rand_map_16, FROZEN_LAKE, i, NO_OF_ITERS)
        # ic(opt_pol)
        toc = time.perf_counter()
        print(f'Took {toc - tic} seconds')
        times.append(toc - tic)
        win_cnt.append(check_policy(rand_map_16, FROZEN_LAKE, opt_pol))

    df_for_vl = pd.DataFrame(data={'Discounts': gammas,
                                   'Wins per thousand runs': win_cnt,
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

    # Do the wins plot
    # Rew vs discount
    sns.lineplot(x='Wins per thousand runs',
                 y='Discounts',
                 data=df_for_vl,
                 palette='pastel')
    sns.set_style('dark')

    plt.title(f'Discounts vs Wins for Frozen Lake Policy Iteration', fontsize=13)
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('outputs/fl_vi_wins_discount.png')

    # Epsilon vs Time plot for ql
    times = []
    win_cnt = []
    # iters = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 1000000]
    epsilon = np.arange(0.05, 1.0, 0.05)
    for i in epsilon:
        tic = time.perf_counter()
        pol, wins = q_learning(FROZEN_LAKE, rand_map_16, epsilon=i)
        # pol = pol_iter(rand_map_16, FROZEN_LAKE, i, NO_OF_ITERS)
        # ic(opt_pol)
        toc = time.perf_counter()
        print(f'Took {toc - tic} seconds')
        times.append(toc - tic)
        win_cnt.append(wins) #check_policy(rand_map_16, FROZEN_LAKE, opt_pol))

    df_for_vl = pd.DataFrame(data={'Epsilon': epsilon,
                                   'Wins per thousand runs': win_cnt,
                                   'Time(seconds)': times})
    ic(df_for_vl.dtypes)
    # Do the time plot
    sns.lineplot(x='Epsilon',
                 y='Time(seconds)',
                 data=df_for_vl,
                 palette='pastel')
    sns.set_style('dark')

    plt.title(f'Time vs Epsilon for Frozen Lake Q-Learning', fontsize=13)
    # plt.legend(loc='upper right')
    plt.show()
    # plt.savefig('outputs/fl_pi_gamma_time.png')

    # Do the wins plot
    # Rew vs time
    sns.lineplot(x='Epsilon',
                 y='Wins per thousand runs',
                 data=df_for_vl,
                 palette='pastel')
    sns.set_style('dark')
    plt.title(f'Wins vs Epsilon for Frozen Lake Q-Learning', fontsize=13)
    # plt.legend(loc='upper right')
    plt.show()



    # Epsilon Decay vs Time plot for ql
    times = []
    win_cnt = []
    # iters = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 1000000]
    epsilon_decay = np.arange(0.05, 1.0, 0.05)
    for i in epsilon:
        tic = time.perf_counter()
        pol, wins = q_learning(FROZEN_LAKE, rand_map_16, epsilon_decay=i)
        # pol = pol_iter(rand_map_16, FROZEN_LAKE, i, NO_OF_ITERS)
        # ic(opt_pol)
        toc = time.perf_counter()
        print(f'Took {toc - tic} seconds')
        times.append(toc - tic)
        win_cnt.append(wins) #check_policy(rand_map_16, FROZEN_LAKE, opt_pol))

    df_for_vl = pd.DataFrame(data={'Epsilon Decay': epsilon_decay,
                                   'Wins per thousand runs': win_cnt,
                                   'Time(seconds)': times})
    ic(df_for_vl.dtypes)
    # Do the time plot
    sns.lineplot(x='Epsilon Decay',
                 y='Time(seconds)',
                 data=df_for_vl,
                 palette='pastel')
    sns.set_style('dark')

    plt.title(f'Time vs Epsilon Decay for Frozen Lake Q-Learning', fontsize=13)
    plt.show()

    # Do the wins plot
    # Rew vs time
    sns.lineplot(x='Epsilon Decay',
                 y='Wins per thousand runs',
                 data=df_for_vl,
                 palette='pastel')
    sns.set_style('dark')
    plt.title(f'Wins vs Epsilon Decay for Frozen Lake Q-Learning', fontsize=13)
    plt.show()



    # Alpha vs Time plot for ql
    times = []
    win_cnt = []
    # iters = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 1000000]
    alpha = np.arange(0.05, 1.0, 0.05)
    for i in epsilon:
        tic = time.perf_counter()
        pol, wins = q_learning(FROZEN_LAKE, rand_map_16, alpha=i)
        # pol = pol_iter(rand_map_16, FROZEN_LAKE, i, NO_OF_ITERS)
        # ic(opt_pol)
        toc = time.perf_counter()
        print(f'Took {toc - tic} seconds')
        times.append(toc - tic)
        win_cnt.append(wins) #check_policy(rand_map_16, FROZEN_LAKE, opt_pol))

    df_for_vl = pd.DataFrame(data={'Alpha': alpha,
                                   'Wins per thousand runs': win_cnt,
                                   'Time(seconds)': times})
    ic(df_for_vl.dtypes)
    # Do the time plot
    sns.lineplot(x='Alpha',
                 y='Time(seconds)',
                 data=df_for_vl,
                 palette='pastel')
    sns.set_style('dark')

    plt.title(f'Time vs Alpha for Frozen Lake Q-Learning', fontsize=13)
    # plt.legend(loc='upper right')
    plt.show()
    # plt.savefig('outputs/fl_pi_gamma_time.png')

    # Do the wins plot
    # Rew vs time
    sns.lineplot(x='Alpha',
                 y='Wins per thousand runs',
                 data=df_for_vl,
                 palette='pastel')
    sns.set_style('dark')
    plt.title(f'Wins vs Alpha for Frozen Lake Q-Learning', fontsize=13)
    # plt.legend(loc='upper right')
    plt.show()


    # Gamma vs Time plot for ql
    times = []
    win_cnt = []
    # iters = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 1000000]
    gamma = np.arange(0.05, 1.0, 0.05)
    for i in epsilon:
        tic = time.perf_counter()
        pol, wins = q_learning(FROZEN_LAKE, rand_map_16, gamma=i)
        # pol = pol_iter(rand_map_16, FROZEN_LAKE, i, NO_OF_ITERS)
        # ic(opt_pol)
        toc = time.perf_counter()
        print(f'Took {toc - tic} seconds')
        times.append(toc - tic)
        win_cnt.append(wins) #check_policy(rand_map_16, FROZEN_LAKE, opt_pol))

    df_for_vl = pd.DataFrame(data={'Gamma': gamma,
                                   'Wins per thousand runs': win_cnt,
                                   'Time(seconds)': times})
    ic(df_for_vl.dtypes)
    # Do the time plot
    sns.lineplot(x='Gamma',
                 y='Time(seconds)',
                 data=df_for_vl,
                 palette='pastel')
    sns.set_style('dark')

    plt.title(f'Time vs Gamma for Frozen Lake Q-Learning', fontsize=13)
    # plt.legend(loc='upper right')
    plt.show()
    # plt.savefig('outputs/fl_pi_gamma_time.png')

    # Do the wins plot
    # Rew vs time
    sns.lineplot(x='Gamma',
                 y='Wins per thousand runs',
                 data=df_for_vl,
                 palette='pastel')
    sns.set_style('dark')
    plt.title(f'Wins vs Gamma for Frozen Lake Q-Learning', fontsize=13)
    # plt.legend(loc='upper right')
    plt.show()

