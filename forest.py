import hiive.mdptoolbox as mdptoolbox
import hiive.mdptoolbox.mdp
import hiive.mdptoolbox.example
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

from timeit import default_timer as timer
from tqdm import tqdm


num_states = 100
P, R = mdptoolbox.example.forest(S=num_states, p=0.01)


def run():
    vi = value_iteration(P, R)
    print(vi.run_stats[-1])
    get_performance_discrete(vi.policy, P, R)

    pi = policy_iteration(P, R)
    print(pi.run_stats[-1])
    get_performance_discrete(pi.policy, P, R)

    ql = q_learning(P, R)
    print(ql.run_stats[-1])
    get_performance_discrete(ql.policy, P, R)


def iteration_plots():
    states = [4, 8, 16, 32, 64, 128, 256, 512]
    vi_iter = []
    pi_iter = []
    vi_time = []
    pi_time = []
    for state in states:
        P, R = mdptoolbox.example.forest(S=state, p=0)
        vi = value_iteration(P, R)
        vi_iter.append(vi.run_stats[-1]['Iteration'])
        vi_time.append(vi.run_stats[-1]['Time'])

        pi = policy_iteration(P, R)
        pi_iter.append(pi.run_stats[-1]['Iteration'])
        pi_time.append(pi.run_stats[-1]['Time'])

    fig, ax = plt.subplots()
    ax.set_title('Forest size vs iterations')
    ax.set_xlabel('forest size')
    ax.set_ylabel('iterations')
    ax.grid(True)
    ax.scatter(states, vi_iter, c='tab:blue', label='value iteration')
    ax.scatter(states, pi_iter, c='tab:orange', label='policy iteration')
    ax.legend(loc='upper left')
    plt.savefig(f'outputs/forest-iterations.png')

    fig, ax = plt.subplots()
    ax.set_title('Forest size vs time')
    ax.set_xlabel('forest size')
    ax.set_ylabel('time (s)')
    ax.grid(True)
    ax.scatter(states, vi_time, c='tab:blue', label='value iteration')
    ax.scatter(states, pi_time, c='tab:orange', label='policy iteration')
    ax.legend(loc='upper left')
    plt.savefig(f'outputs/forest-time.png')


def value_iteration(P, R):
    np.random.seed(1337)
    print('Value iterating')
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.99)
    # vi.setVerbose()
    start = timer()
    vi.run()
    end = timer()
    print(end - start)
    return vi


def policy_iteration(P, R):
    np.random.seed(1337)
    print('Policy iterating')
    pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.99, max_iter=1000)
    # pi.setVerbose()
    start = timer()
    pi.run()
    end = timer()
    print(end - start)
    return pi


def q_learning(P, R, epsilon_min=0.1, epsilon_decay=0.999):
    np.random.seed(1337)
    print('Q Learning')
    ql = mdptoolbox.mdp.QLearning(P, R, 0.99,
                                  n_iter=20000,
                                  alpha_decay=0.999,
                                  alpha_min=0.01,
                                  epsilon_min=epsilon_min,
                                  epsilon_decay=epsilon_decay)
    # ql.setVerbose()
    start = timer()
    ql.run()
    end = timer()
    print(end - start)
    return ql


def get_performance_discrete(policy, P, R, num_trials=100):
    np.random.seed(1337)
    print(policy)
    results = []
    for k in range(num_trials):
        total_reward = 0
        state = 0
        for step in range(100):
            action = policy[state]
            total_reward += R[state, action]
            state = np.random.choice(range(num_states), p=P[action, state])
            if state == 0:
                break

        results.append(total_reward)

    print(f'Average total reward (discrete): {np.mean(results)}')
    return np.mean(results)

if __name__ == '__main__':
    ql = q_learning(P, R)
    print(ql.run_stats[-1])
    print(ql.time)
    get_performance_discrete(ql.policy, P, R)
