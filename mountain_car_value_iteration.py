import gymnasium as gym
import numpy as np

from mountain_car_base import build_custom_p_matrix, test


def value_iteration(env, P, num_states, num_iterations=5000, gamma=0.99, threshold=1e-20):
    value_table = np.zeros(num_states)

    for i in range(num_iterations):
        updated_value_table = np.copy(value_table)

        for s in range(num_states):
            Q_values = [sum([(r + gamma * updated_value_table[s_])
                             for s_, r, done in P[s][a]])
                        for a in range(env.action_space.n)]

            value_table[s] = max(Q_values)

        if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
            break
    return value_table


def extract_policy(P, value_table, num_states, gamma=0.99):
    policy = np.zeros(num_states)
    for s in range(num_states):
        Q_values = [sum([(r + gamma * value_table[s_])
                         for s_, r, done in P[s][a]])
                    for a in range(3)]
        policy[s] = np.argmax(np.array(Q_values))
    return policy


if __name__ == '__main__':
    env = gym.make('MountainCar-v0', max_episode_steps=1000)

    n_bins = 25
    num_states = n_bins * n_bins
    pos_bins = np.linspace(env.observation_space.low[0], env.observation_space.high[0], n_bins)
    vel_bins = np.linspace(env.observation_space.low[1], env.observation_space.high[1], n_bins)

    P_custom = build_custom_p_matrix(env, n_bins, pos_bins, vel_bins)

    print("\n--- Value Iteration ---")
    v_opt = value_iteration(env, P_custom, num_states)
    optimal_policy = extract_policy(P_custom, v_opt, num_states)
    print(optimal_policy)

    print("\n--- Testing ---")
    steps_list = []
    for i in range(200):
        reward, steps, success = test(env, optimal_policy, n_bins, pos_bins, vel_bins)
        steps_list.append(steps)
        print(f"Test {i+1}: Success={success}, Steps={steps}")

    print(f"Maximum steps: {np.max(steps_list)}, Minimum steps: {np.min(steps_list)}, Average steps: {np.mean(steps_list)}")

    print("\n--- Demonstration ---")
    env = gym.make('MountainCar-v0', render_mode="human", max_episode_steps=1000)
    reward, steps, success = test(env, optimal_policy, n_bins, pos_bins, vel_bins)
    print(f"Success={success}, Steps={steps}")
    env.close()