import gymnasium as gym
import numpy as np


def get_state_index(state, n_bins, pos_bins, vel_bins):
    p_idx = np.digitize(state[0], pos_bins) - 1
    v_idx = np.digitize(state[1], vel_bins) - 1
    p_idx = min(n_bins - 1, max(0, p_idx))
    v_idx = min(n_bins - 1, max(0, v_idx))
    return p_idx * n_bins + v_idx


def calculate_custom_reward(reward, pos, vel, action, next_pos, next_vel):
    # чим правіше - тим краще
    reward += next_pos

    # нагорода за газ вліво при від'ємній швидкості або газ вправо при додатній
    if (next_vel > 0 and action == 2) or (next_vel < 0 and action == 0):
        reward += abs(next_vel)

    # штраф за зупинку або низьку активність далеко від цілі
    if abs(next_vel) < 0.001 and next_pos < 0.4:
        reward -= 1.0

    return reward


def build_custom_p_matrix(env, n_bins, pos_bins, vel_bins):
    env.reset()
    num_states = n_bins * n_bins

    # Створюємо словник P, де для кожного стану 's' та кожної дії 'a' буде зберігатися список переходів і винагород.
    P = {s: {a: [] for a in range(env.action_space.n)} for s in range(num_states)}

    # по всіх значеннях позиції (від -1.2 до 0.6).
    for p in range(n_bins):
        # по всіх значеннях швидкості (від -0.07 до 0.07).
        for v in range(n_bins):
            state_idx = p * n_bins + v

            # Для кожного стану перевіряємо результати всіх 3-х дій (0, 1, 2).
            for a in range(env.action_space.n):
                curr_pos, curr_vel = pos_bins[p], vel_bins[v]

                # значення пточного стану у внутрішній стан середовища.
                env.unwrapped.state = np.array([curr_pos, curr_vel])

                next_obs, reward, terminated, truncated, _ = env.step(a)  # новий стан після кроку з поточною дією

                next_s_idx = get_state_index(next_obs, n_bins, pos_bins, vel_bins) # дискретний індекс нового стану

                # винагорода на основі позиції і напрямку швидкості
                r = calculate_custom_reward(reward, curr_pos, curr_vel, a, next_obs[0], next_obs[1])
                done = terminated or truncated

                # результат у матрицю P, prob = 1.0, бо Mountain Car на 100% детермінований
                # Формат: (наступний_стан, винагорода, done).
                P[state_idx][a].append((next_s_idx, r, done))
    return P


def test(env, optimal_policy, n_bins, pos_bins, vel_bins):
    obs, _ = env.reset()

    total_reward = 0
    done = False
    step = 0
    while not done:
        s_idx = get_state_index(env.unwrapped.state, n_bins, pos_bins, vel_bins)
        action = int(optimal_policy[s_idx])
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        step += 1
        done = term or trunc

    return total_reward, step + 1, obs[0] >= 0.5
