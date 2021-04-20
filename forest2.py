import hiive.mdptoolbox as mdptoolbox
import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt

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

def forest_val_iter(states, seed=42, sparse=False, gamma=GAMMA):
    """Runs the value iteration algorithm on the forest example.

    :param states: No. of states or years to run the simulation.
    :param seed: Random seed used to generate the forest
    :param sparse: "If true, the matrices will be returned in their sparse
    format.
    :return: val_iter which contains P and R
    """
    np.random.seed(seed)
    P, R = mdptoolbox.example.forest(S=states, p=PROB_WILDFIRE)
    print(f'Beginning policy iteration for the forest')
    val_iter = mdptoolbox.mdp.PolicyIteration(P, R, gamma, max_iter=MAX_ITERS_VAL)
    print(f'Ending policy iteration for the forest')

    return val_iter

def forest_pol_iter(states, seed=42, sparse=False, gamma=GAMMA):
    """Runs the policy iteration algorithm on the forest example.

    :param states: No. of states or years to run the simulation.
    :param seed: Random seed used to generate the forest
    :param sparse: "If true, the matrices will be returned in their sparse
    format.
    :return: val_iter which contains P and R
    """
    np.random.seed(seed)
    P, R = mdptoolbox.example.forest(S=states, p=PROB_WILDFIRE)
    print(f'Beginning value iteration for the forest')
    pol_iter = mdptoolbox.mdp.ValueIteration(P, R, gamma, max_iter=MAX_ITERS_POL)
    print(f'Ending value iteration for the forest')

    return pol_iter

def forest_q_learning(states, seed=42, sparse=False, gamma=GAMMA):
    """Runs the q_learning algorithm on the forest example.

    :param states: No. of states or years to run the simulation.
    :param seed: Random seed used to generate the forest
    :param sparse: "If true, the matrices will be returned in their sparse
    format.
    :return: val_iter which contains P and R
    """
    np.random.seed(seed)
    P, R = mdptoolbox.example.forest(S=states, p=PROB_WILDFIRE)
    print(f'Beginning Q-learning for the forest')
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
                     states,
                     seed)
    print('Done')
    return ql, perf

def find_perf(P, R, policy, no_states, seed=42, exercises=100):
    """Finds the average reward for the policy.

    :param P: Transitions
    :param R: Rewards
    :param policy: The policy to be tested out.
    :param no_states: Number of states in problem.
    :param seed: Random seed used to generate the forest
    :param tries: The number of trials
    :return: Average reward for the policy.
    """
    np.random.seed(seed)
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

if __name__ == '__main__':
    result, perf = forest_q_learning(100)
    print(f'Time for Q-Learning: {result.time}')
    print(f'Error for Q-Learning: {result.error_mean}')
    print(f'Run Stats\n---------\n{result.run_stats[-1]}')
    print(f'Average reward:{perf}')
