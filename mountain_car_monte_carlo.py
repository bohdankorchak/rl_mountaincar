import gymnasium as gym
import numpy as np
from collections import defaultdict

from mountain_car_base import get_state_index, calculate_custom_reward, test


def generate_episode(env, policy, epsilon, n_bins, pos_bins, vel_bins):
    episode = []  # Список для запису історії: (стан, дія, винагорода)
    state_obs, _ = env.reset()

    env.unwrapped.state = np.array([np.random.uniform(-0.6, -0.4), 0.0])
    state_obs = env.unwrapped.state

    done = False
    while not done:
        s_idx = get_state_index(state_obs, n_bins, pos_bins, vel_bins)

        # epsilon-greedy вибір дії
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        # інакше найкращу дію з поточної полісі
        else:
            action = int(policy[s_idx])

        next_obs, reward, terminated, truncated, _ = env.step(action) # виконуємо обрану дію

        # кастомна винагороду за цей крок
        reward = calculate_custom_reward(reward, state_obs[0], state_obs[1], action, next_obs[0], next_obs[1])

        # Записуємо кроку в епізод
        episode.append((s_idx, action, reward))

        state_obs = next_obs  # Переходимо до наступного стану
        done = terminated or truncated  # Перевіряємо, чи кінець заїзду

    return episode  # Повертаємо всю історію заїзду


def monte_carlo(env, n_bins, pos_bins, vel_bins, num_episodes=35000, epsilon=0.5):
    num_states = n_bins * n_bins

    Q = defaultdict(lambda: np.zeros(env.action_space.n))  # оцінка корисності дії в стані
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n)) # cума винагород для кожної пари стан, дія
    returns_count = defaultdict(lambda: np.zeros(env.action_space.n)) # кількість відвідування пари (стан, дія)

    # початкова стратегія: завжди тиснути вправо (дія 2)
    policy = np.full(num_states, 2)

    for i in range(num_episodes):

        curr_epsilon = max(0.05, epsilon * (1 - i / num_episodes)) # на кожному кроці зменшуємо epsilon, для зменшення рандомних дій

        episode = generate_episode(env, policy, curr_epsilon, n_bins, pos_bins, vel_bins)

        all_state_action_pairs = [(s, a) for (s, a, r) in episode]
        rewards = [r for (s, a, r) in episode]

        for t, (s_idx, action, reward) in enumerate(episode):
            if not (s_idx, action) in all_state_action_pairs[0:t]:

                R = sum(rewards[t:])

                returns_sum[s_idx][action] = returns_sum[s_idx][action] + R
                returns_count[s_idx][action] += 1

                # оновлюємо середнє значення Q для цієї дії в цьому стані
                Q[s_idx][action] = returns_sum[s_idx][action] / returns_count[s_idx][action]

                # оновлюємо полісі:
                # у поточному стані обирається дія, з найкращим середнім результатом
                policy[s_idx] = np.argmax(Q[s_idx])

        if (i + 1) % 1000 == 0:
            print(f"Episode {i + 1}/{num_episodes}. States covered: {len(Q)}/{num_states}")

    return policy


if __name__ == '__main__':
    env = gym.make('MountainCar-v0', max_episode_steps=2000)

    n_bins = 25
    num_states = n_bins * n_bins
    pos_bins = np.linspace(env.observation_space.low[0], env.observation_space.high[0], n_bins)
    vel_bins = np.linspace(env.observation_space.low[1], env.observation_space.high[1], n_bins)

    print("\n--- Monte Carlo ---")
    optimal_policy = monte_carlo(env, n_bins, pos_bins, vel_bins)

    print(optimal_policy)

    print("\n--- Testing ---")
    steps_list = []
    for i in range(200):
        reward, steps, success = test(env, optimal_policy, n_bins, pos_bins, vel_bins)
        steps_list.append(steps)
        print(f"Test {i + 1}: Success={success}, Steps={steps}")
    print(f"Maximum steps: {np.max(steps_list)}, Minimum steps: {np.min(steps_list)}, Average steps: {np.mean(steps_list)}")

    print("\n--- Demonstration ---")
    env = gym.make('MountainCar-v0', render_mode="human", max_episode_steps=1000)
    reward, steps, success = test(env, optimal_policy, n_bins, pos_bins, vel_bins)
    print(f"Success={success}, Steps={steps}")
    env.close()
