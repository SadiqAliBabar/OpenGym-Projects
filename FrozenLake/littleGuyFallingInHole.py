
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import  pickle

def run(episodes, is_slippery, is_training, rendering= False):

    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=is_slippery , render_mode = 'human' if rendering else None)

    if(is_training):
        q_table = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
        try:
            with open('frozen_lake8x8.pkl', 'rb') as f:
                q_table = pickle.load(f)  # Use q_table instead of q
        except Exception as e:
            print('Model file not found, initializing new Q-table.')
            q_table = np.zeros((env.observation_space.n, env.action_space.n))


    # intializing the hyperparemeter
    alpha = 0.1
    gamma = 0.95
    epsilon = 1
    epsilon_decay_rate = 0.9999
    epsilon_min = 0.01
    episodes = episodes
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for episode in range(episodes):
        state, _ = env.reset()
        truncated, terminated = False,False

        while not (truncated or terminated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            reward = -0.01 if not terminated else reward  # Encourage movement
            next_state, reward, terminated, truncated, _ = env.step(action)

            if(is_training):
                old_qvalue = q_table[state,action]
                best_qvalues = np.max(q_table[next_state,:])
                q_table[state, action] = (1 - alpha) * old_qvalue + alpha *(reward + gamma * best_qvalues)

            state = next_state

        epsilon = max(epsilon_min, epsilon_decay_rate*epsilon)

        if (epsilon == 0):
            alpha = 0.0001

        if (reward==1):
            rewards_per_episode[episode] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')

    if is_training:
        f = open("frozen_lake8x8.pkl", "wb")
        pickle.dump(q_table, f)
        f.close()


if __name__ == '__main__':
    # run(15000)

    # run(10000, is_training=True ,rendering=False,is_slippery=True)
    run(10, is_training=False ,rendering=True,is_slippery=True)








