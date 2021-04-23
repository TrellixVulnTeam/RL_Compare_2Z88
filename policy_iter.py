from icecream import ic
import gym
import numpy as np

import time

# Constants
GAMMA = 1.0
LIMIT = 1e-10
NO_OF_ITERS = 200000
FROZEN_LAKE = 'FrozenLake-v0'
SR_TRANS_PROB = 0
SR_NEXT_STATE = 1
SR_REW_PROB = 2

def val_iter(gamma, environ_name):
    """Value iteration

    :param gamma: Discount factor from 0 to 1
    :param environ_name: Name used in Open AI gym
    :return: Latest value table
    """
    environ = gym.make(environ_name)
    no_of_states = environ.observation_space.n
    no_of_actions = environ.action_space.n
    val_table = np.zeros(no_of_states)
    print(f'Running value iteration for {environ_name}')
    ## Loop through the iterations set
    for iters in range(NO_OF_ITERS):
        if iters % 10000 == 0:
            ic(iters)
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

                    rewards.append((state_reward[SR_TRANS_PROB] * \
                                    (state_reward[SR_REW_PROB] + gamma * \
                                    new_val_table[SR_NEXT_STATE])))
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
    print(f'Converged after {iters} iteraions.')
    return val_table


def find_best_policy(gamma, val_table, environ_name):
    """Use the value table to find the best policy

    :param gamma: Discount factor from 0 to 1
    :param val_table: The table to search
    :return: The best policy found from the value table
    """
    environ = gym.make(environ_name)
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
    tic = time.perf_counter()
    opt_pol = find_best_policy(GAMMA,
                               val_iter(GAMMA, FROZEN_LAKE),
                               FROZEN_LAKE)
    ic(opt_pol)
    print(f'Took {time.perf_counter() - tic} seconds')
