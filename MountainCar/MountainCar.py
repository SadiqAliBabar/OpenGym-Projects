import pickle
import  gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def run(episodes, is_training, render = False):
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    # dividing the position and the Velocity
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)


    if (is_training):
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
    else:
        with open('mountain_car.pkl', 'rb') as f:
            q = pickle.load(f)

    alpha = 0.9
    gamma = 0.9
    epsilon = 1
    epsilon_decay_rate = 2/episodes
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)


    for episode in range(episodes):
        state, _ = env.reset()
        state_p = np.clip(np.digitize(state[0], pos_space) - 1, 0, len(pos_space) - 1)
        state_v = np.clip(np.digitize(state[1], vel_space) - 1, 0, len(vel_space) - 1)

        terminated = False  # this will gonna true when reaching the goal
        rewards = 0

        while not terminated:

            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p,state_v,:])

            new_state, reward, terminated, _, _ = env.step(action)

            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training:
                old_qvalue = q[state_p,state_v,action]
                best_qvalue =  np.max(q[new_state_p, new_state_v,:])
                q[state_p, state_v, action] = (1-alpha)*old_qvalue + alpha*(reward+gamma*best_qvalue)



            state = new_state
            state_p = new_state_p
            state_v = new_state_v

            rewards+=reward
        epsilon = max(epsilon * 0.99, 0.01)
        rewards_per_episode[episode]= rewards
    env.close()

    if is_training:
        f = open('mountain_car.pkl','wb')
        pickle.dump(q,f)
        f.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t - 100):(t + 1)])
    plt.plot(mean_rewards)
    plt.savefig(f'mountain_car.png')

if __name__ == "__main__":
    # run(5000, is_training=True, render=False)
    run(5, is_training=False, render=True)



