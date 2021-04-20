from icecream import ic
import gym
import numpy as np

import time

# Constants
GAMMA = 0.99
LIMIT = 1e-15
NO_OF_ITERS = 200000
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

def val_iter(gamma, environ_name):
    """Value iteration

    :param gamma: Discount factor from 0 to 1
    :param environ_name: Name used in Open AI gym
    :return: Latest value table
    """
    # environ = gym.make(environ_name)
    from gym.envs.toy_text.frozen_lake import generate_random_map
    random_map = generate_random_map(size=20, p=0.8)

    environ = gym.make(environ_name, desc=random_map)

    # environ = gym.make(environ_name, map_name="20x10")
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



if __name__ == '__main__':
    env_choice, alg_choice = '', ''
    # Ask user which environment/algorithm
    env_choice = input('1. Frozen Lake, 2. nChain or 3. Run Everything')
    if env_choice != '3':
        alg_choice = input('1. Value, 2. Policy or 3. Q-learning')

    tic = time.perf_counter()
    if env_choice == '1':
        if alg_choice == '1':

            opt_pol = find_best_policy(GAMMA,
                                       val_iter(GAMMA, FROZEN_LAKE),
                                       FROZEN_LAKE)
            print(f'Optimum policy for frozen lake...\n\n{opt_pol}')
        elif alg_choice == '2':
            print('Not yet implemented')
        elif alg_choice == '3':
            print(f'Q-table\n\n{q_learning(GAMMA, LEARN_RATE, FROZEN_LAKE)}')

    elif env_choice == '2':
        if alg_choice == '3':
            print(f'Q-table\n\n{q_learning(GAMMA, LEARN_RATE, BLACKJACK)}')

    print(f'Took {time.perf_counter() - tic} seconds')
