import hiive.mdptoolbox as mdptoolbox
import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp
from icecream import ic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import time


# Constants
PROB_WILDFIRE = 0.9
MAX_ITERS_VAL = 1500
MAX_ITERS_POL = 1500
GAMMA = 0.99
N_ITERS_QL = 20000
ALPHA_MIN = 0.01
EPSILON_MIN = 0.1
ALPHA_DECAY = 0.999
EPSILON_DECAY = 0.999
SEED = 42
STATES_TO_TEST = 640

def forest_val_iter(sparse=False, gamma=GAMMA):
    """Runs the value iteration algorithm on the forest example.

    :param states: No. of states or years to run the simulation.
    :param seed: Random seed used to generate the forest
    :param sparse: "If true, the matrices will be returned in their sparse
    format.
    :return: val_iter which contains P and R
    """
    P, R = mdptoolbox.example.forest(S=states, p=PROB_WILDFIRE)
    print(f'Beginning policy iteration for the forest for {states} states')
    val_iter = mdptoolbox.mdp.PolicyIteration(P, R, gamma, max_iter=MAX_ITERS_VAL)
    print(f'Ending policy iteration for the forest')

    return val_iter

def forest_pol_iter(sparse=False, gamma=GAMMA):
    """Runs the policy iteration algorithm on the forest example.

    :param states: No. of states or years to run the simulation.
    :param seed: Random seed used to generate the forest
    :param sparse: "If true, the matrices will be returned in their sparse
    format.
    :return: val_iter which contains P and R
    """
    # np.random.seed(seed)
    print(f'Beginning value iteration for the forest for {states} states')
    pol_iter = mdptoolbox.mdp.ValueIteration(P, R, gamma, max_iter=MAX_ITERS_POL)
    print(f'Ending value iteration for the forest')

    return pol_iter

def forest_q_learning(P, R, states=10, sparse=False, gamma=GAMMA):
    """Runs the q_learning algorithm on the forest example.

    :param states: No. of states or years to run the simulation.
    :param seed: Random seed used to generate the forest
    :param sparse: "If true, the matrices will be returned in their sparse
    format.
    :return: val_iter which contains P and R
    """
    # np.random.seed(seed)
    # P, R = mdptoolbox.example.forest(S=states, p=PROB_WILDFIRE)
    print(f'Beginning Q-learning for the forest for {states} states')
    ql = mdptoolbox.mdp.QLearning(P,
                                  R,
                                  gamma,
                                  n_iter=N_ITERS_QL,
                                  alpha_min=ALPHA_MIN,
                                  epsilon_min=EPSILON_MIN,
                                  alpha_decay=ALPHA_DECAY,
                                  epsilon_decay=EPSILON_DECAY)
    ql.run()
    print(f'Ending Q-learning for the forest')
    perf = find_perf(P,
                     R,
                     ql.policy,
                     states)

    print('Done')
    return ql, perf

def find_perf(P, R, policy, no_states, exercises=100):
    """Finds the average reward for the policy.

    :param P: Transitions
    :param R: Rewards
    :param policy: The policy to be tested out.
    :param no_states: Number of states in problem.
    :param seed: Random seed used to generate the forest
    :param tries: The number of trials
    :return: Average reward for the policy.
    """
    # np.random.seed(seed)
    print('Finding the average reward...')
    totals = []
    for _ in range(exercises):
        total, state = 0, 0
        for _ in range(100):
            action = policy[state]
            total += R[state, action]
            probability = P[action, state]
            state = np.random.choice(range(no_states),
                                     p=probability)
            if not state:
                break
        totals.append(total)
    result = np.mean(totals)

    return result

def build_forest_charts(exercises=100):
    """Build various charts for each of the algorithms for the forest MDP.

    :param P: Transitions
    :param R: Rewards
    :param seed: Random seed used to generate the forest
    :param tries: The number of trials
    :return:
    """
    # How many states to use for states and times vs rewards
    chart_data = np.array([10, 20, 40, 80, 160, 320, 640])
    iteration_data = []
    perf_data = []
    time_data = []
    for state in chart_data:
        np.random.seed(SEED)
        P, R = mdptoolbox.example.forest(S=state, p=PROB_WILDFIRE)
        tic = time.perf_counter()
        val_res, perf = forest_q_learning(P, R, state)
        toc = time.perf_counter()
        iteration_data.append(val_res.run_stats[-1]['Iteration'])
        perf_data.append(perf)
        time_data.append(toc-tic)

    df_for_ql = pd.DataFrame(data=[chart_data, iteration_data, perf_data, time_data]).transpose()
    df_for_ql.columns=['Forest Size', 'Iterations', 'Rewards', 'Time(seconds)']
                            #  np.hstack((iteration_data[:, None],
                            #             perf_data[:, None],
                            #             time_data[:, None])))

    sns.lineplot(x='Forest Size', y='Rewards', data=df_for_ql)
    plt.show()
    sns.lineplot(x='Forest Size', y='Time(seconds)', data=df_for_ql)
    plt.show()
    sns.lineplot(x='Forest Size', y='Iterations', data=df_for_ql)
    plt.show()
    ic(chart_data)



if __name__ == '__main__':
    # np.random.seed(SEED)
    # P, R = mdptoolbox.example.forest(S=STATES_TO_TEST, p=PROB_WILDFIRE)
    # result, perf = forest_q_learning(100)
    # print(f'Time for Q-Learning: {result.time}')
    # print(f'Error for Q-Learning: {result.error_mean}')
    # print(f'Run Stats\n---------\n{result.run_stats[-1]}')
    # print(f'Average reward:{perf}')
    build_forest_charts()
