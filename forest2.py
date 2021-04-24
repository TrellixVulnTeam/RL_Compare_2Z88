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
PROB_WILDFIRE = 0.01
MAX_ITERS_VAL = 1500
MAX_ITERS_POL = 1500
GAMMA = 0.99
N_ITERS_QL = 20000
ALPHA_MIN = 0.01
EPSILON_MIN = 0.1
ALPHA_DECAY = 0.999
EPSILON_DECAY = 0.999
EPSILON=0.1
SEED = 42
STATES_TO_TEST = 640

def forest_val_iter(P, R, states, sparse=False, gamma=GAMMA, charts=False):
    """Runs the value iteration algorithm on the forest example.

    :param P: Transitions
    :param R: Rewards
    :param states: No. of states or years to run the simulation.
    :param seed: Random seed used to generate the forest
    :param sparse: "If true, the matrices will be returned in their sparse
    format.
    :return: val_iter which contains P and R
    """
    P, R = mdptoolbox.example.forest(S=states, p=PROB_WILDFIRE)
    print(f'Beginning policy iteration for the forest for {states} states')
    val_iter = mdptoolbox.mdp.PolicyIteration(P, R, gamma, max_iter=MAX_ITERS_VAL)
    val_iter.run()
    print(f'Ending policy iteration for the forest')
    achieve = find_achieve(P,
                           R,
                           val_iter.policy,
                           states)


    return val_iter, achieve

    return val_iter

def forest_pol_iter(P, R, states=10, sparse=False, gamma=GAMMA):
    """Runs the policy iteration algorithm on the forest example.

    :param P: Transitions
    :param R: Rewards
    :param states: No. of states or years to run the simulation.
    :param seed: Random seed used to generate the forest
    :param sparse: "If true, the matrices will be returned in their sparse
    format.
    :return: val_iter which contains P and R
    """
    # np.random.seed(seed)
    print(f'Beginning value iteration for the forest for {states} states')
    pol_iter = mdptoolbox.mdp.ValueIteration(P, R, gamma, max_iter=MAX_ITERS_POL)
    pol_iter.run()
    print(f'Ending value iteration for the forest')
    achieve = find_achieve(P,
                           R,
                           pol_iter.policy,
                           states)

    return pol_iter, achieve

def forest_q_learning(P, R, states=10, sparse=False, gamma=GAMMA,
                      epsilon=EPSILON,
                      epsilon_decay=EPSILON_DECAY,
                      alpha=0.01):
    """Runs the q_learning algorithm on the forest example.

    :param P: Transitions
    :param R: Rewards
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
                                  epsilon_decay=epsilon_decay,
                                  epsilon=epsilon,
                                  alpha=alpha)
    ql.run()
    print(f'Ending Q-learning for the forest')
    achieve = find_achieve(P,
                           R,
                           ql.policy,
                           states)

    print('Done')
    return ql, achieve

def find_achieve(P, R, policy, no_states, exercises=100):
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

def build_forest_iter_charts(exercises=100):
    """Build various charts for each of the iteration algorithms for the
    forest MDP.

    :param tries: The number of trials
    :return: None
    """
    # How many states to use for states and times vs rewards
    states = [2, 5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 400, 500, 750, 1000]

    # Create data for value iteration
    iteration_data = []
    achieve_data = []
    time_data = []
    alg_name_data = []
    states_data=[]

    # Loop through both value and policy iteration algorithms
    for algorithm, alg_name in [(forest_val_iter, 'Value Iteration'),
                                (forest_pol_iter, 'Policy Iteration')]:
        # Loop through different states running value iteration for each and
        # grabbing the different statistics.
        for state in states:
            np.random.seed(SEED)
            P, R = mdptoolbox.example.forest(S=state, p=PROB_WILDFIRE)
            tic = time.perf_counter()
            result, achieve = algorithm(P, R, state)
            toc = time.perf_counter()
            iteration_data.append(int(result.run_stats[-1]['Iteration']))
            achieve_data.append(float(achieve))
            time_data.append(float(toc-tic))
            alg_name_data.append(alg_name)
            states_data.append(state)
            # alg_name_data.append(alg_name==forest_val_iter)
        print(f'Results for {alg_name} Iteration\n----------------------\n\n')
        print(result.run_stats[-1])
        print(f'Average Total Reward = {achieve}')

        df_for_vl = pd.DataFrame(data={'Forest Size': states_data,
                                       'Iterations': iteration_data,
                                       'Rewards': achieve_data,
                                       'Time(seconds)': time_data,
                                       'Algorithm': alg_name_data})
    ic(df_for_vl.dtypes)

    sns.lineplot(x='Time(seconds)',
                 y='Iterations',
                 data=df_for_vl,
                 hue='Algorithm',
                 palette='pastel')
    sns.set_style('dark')

    plt.title(f'Forest Size vs Reward', fontsize=13)
    plt.legend(loc='upper right')
    plt.show()

    ic(states)


def build_forest_ql_charts(var_min,
                           var_max,
                           var_step,
                           var_name,
                           exercises=100):
    """Build various charts for each of the iteration algorithms for the
    forest MDP. Includes:

    1. Reward vs. Epsilon
    2. Reward vs. Epsilon Decay
    3. Rewards vs. Alpha
    1. Time vs. Epsilon
    2. Time vs. Epsilon Decay
    3. Time vs. Alpha


    :param tries: The number of trials
    :return: None
    """
    # How many states to use for states and times vs rewards
    states = [2, 5, 10, 20, 30] #40, 50, 75, 100, 200, 300, 400, 500, 750, 1000]

    # Create data for value iteration
    iteration_data = []
    achieve_data = []
    time_data = []
    var_data = []
    states_data=[]

    # Loop through different states running value iteration for each and
    # grabbing the different statistics.
    # for epsilon in []
    for var_cnt in np.arange(var_min, var_max, var_step):
        np.random.seed(SEED)
        P, R = mdptoolbox.example.forest(S=STATES_TO_TEST, p=PROB_WILDFIRE)
        tic = time.perf_counter()
        result, achieve = forest_q_learning(P, R, STATES_TO_TEST)#, epsilon)
        toc = time.perf_counter()
        print(result.run_stats[-1])
        iteration_data.append(int(result.run_stats[-1]['Iteration']))
        achieve_data.append(float(achieve))
        time_data.append(float(toc-tic))
        var_data.append(var_cnt)
        # states_data.append(state)
        # alg_name_data.append(alg_name==forest_val_iter)
    print(f'Results for  Q-Learning\n----------------------\n\n')
    print(result.run_stats[-1])
    print(f'Average Total Reward = {achieve}')

    df_for_vl = pd.DataFrame(data={'Iterations': iteration_data,
                                   'Rewards': achieve_data,
                                   'Time(seconds)': time_data,
                                   var_name: var_data})
    ic(df_for_vl.dtypes)
    # Do the time plot
    sns.lineplot(x=var_name,
                 y='Time(seconds)',
                 data=df_for_vl,
                 palette='pastel')
    sns.set_style('dark')

    plt.title(f'Time vs {var_name} for Forest Q-Learning', fontsize=13)
    plt.legend(loc='upper right')
    plt.show()

    # Do the reward plot
    sns.lineplot(x=var_name,
                 y='Rewards',
                 data=df_for_vl,
                 palette='pastel')
    sns.set_style('dark')

    plt.title(f'Reward vs {var_name} for Forest Q-Learning', fontsize=13)
    y = var_name,
    plt.legend(loc='upper right')
    plt.show()
    ic(states)






if __name__ == '__main__':
    # np.random.seed(SEED)
    # P, R = mdptoolbox.example.forest(S=STATES_TO_TEST, p=PROB_WILDFIRE)
    # result, perf = forest_q_learning(100)
    # print(f'Time for Q-Learning: {result.time}')
    # print(f'Error for Q-Learning: {result.error_mean}')
    # print(f'Run Stats\n---------\n{result.run_stats[-1]}')
    # print(f'Average reward:{perf}')


    build_forest_iter_charts()

    # build_forest_ql_charts(0.1, 0.9, 0.05, 'epsilon')
    # build_forest_ql_charts(0, 0.9, 0.02, 'alpha')
    # build_forest_ql_charts(0, 1.0, 0.02, 'epsilon_decay')
    # build_forest_ql_charts(0, 1.0, 0.02, 'gamma')
