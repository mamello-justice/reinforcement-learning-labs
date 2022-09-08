import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def random_argmax(array):
    max_value = np.max(array)
    max_indices = np.where(array == max_value)[0]
    return np.random.choice(max_indices)


def epsilon_greedy_action(Q, state, epsilon):
    action_values = Q[state]

    if np.random.rand() < epsilon:
        return np.random.randint(0, len(action_values))

    return random_argmax(action_values)


def sarsa(env: gym.Env, _lambda, episodes, gamma=1, alpha=0.5, epsilon=0.1, seed=None):
    shape = (env.observation_space.n, env.action_space.n)
    
    Q_max_state = []

    # Initialize Q & e
    Q = np.full(shape, 10, dtype=np.float64)
    e = np.zeros(shape, dtype=np.float64)

    # For each episode
    for ep in trange(episodes, desc=f"λ={_lambda}"):
        # Initialize state and action
        state = env.reset(seed=seed)
        action = epsilon_greedy_action(Q, state, epsilon)

        while True:
            # Take action and observe next state (state_prime) and reward
            state_prime, reward, terminated, _ = env.step(action)

            # Choose next action using epsilon greedy
            action_prime = epsilon_greedy_action(Q, state_prime, epsilon)

            sigma = reward + gamma * Q[state_prime,
                                       action_prime] - Q[state, action]
            e[state, action] += 1

            for s in range(env.observation_space.n):
                for a in range(env.action_space.n):
                    Q[s, a] += alpha * sigma * e[s, a]
                    e[s, a] *= gamma * _lambda

            state = state_prime
            action = action_prime

            if terminated:
                break
            
        # Append max Q value
        Q_max_state.append(Q.max(axis=1))

    # DUMP DUMP DUMP
    np.savetxt(f"./data/Q_lambda_{_lambda}.csv", Q_max_state, delimiter=",")
        

    return Q


def main():
    env = gym.make('CliffWalking-v0')

    lambdas = [0.0, 0.3, 0.5, 0.7, 0.9]
    episodes = 500

    for _lambda in lambdas:
        sarsa(env, _lambda, episodes)

    env.close()

if __name__ == '__main__':
    main()
