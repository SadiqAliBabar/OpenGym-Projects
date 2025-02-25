import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt


def run(episodes, is_training, rendering=False):
    env = gym.make('Taxi-v3', render_mode='human' if rendering else None)

    if is_training:
        q_table = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        try:
            with open('taxi_q_table.pkl', 'rb') as f:
                q_table = pickle.load(f)
        except FileNotFoundError:
            print('Model file not found, initializing new Q-table.')
            q_table = np.zeros((env.observation_space.n, env.action_space.n))

    alpha = 0.9
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay_rate = 0.9995
    epsilon_minimum = 0.01
    max_steps_per_episode = 100

    rewards_per_episode = np.zeros(episodes)

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_rewards = 0

        for step in range(max_steps_per_episode):
            if is_training and np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            next_state, reward, done, truncated, _ = env.step(action)

            if is_training:
                old_qvalue = q_table[state, action]
                max_next_q = np.max(q_table[next_state, :])
                q_table[state, action] = (1 - alpha) * old_qvalue + alpha * (reward + gamma * max_next_q)

            state = next_state
            total_rewards += reward

            if done or truncated:
                break

        epsilon = max(epsilon * epsilon_decay_rate, epsilon_minimum)
        rewards_per_episode[episode] = total_rewards

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])
    plt.plot(sum_rewards)
    plt.savefig('taxi_q_learning.png')

    if is_training:
        with open("taxi_q_table.pkl", "wb") as f:
            pickle.dump(q_table, f)


if __name__ == "__main__":
    # run(10000, is_training=True, rendering=False)
    run(5, is_training=False, rendering=True)
