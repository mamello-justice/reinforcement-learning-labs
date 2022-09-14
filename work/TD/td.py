import argparse, os, subprocess
import gym
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import trange


# Mamello:1851317


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

            for t in range(env.observation_space.n):
                for a in range(env.action_space.n):
                    Q[t, a] += alpha * sigma * e[t, a]
                    e[t, a] *= gamma * _lambda

            state = state_prime
            action = action_prime

            if terminated:
                break
            
        # Append max Q value
        Q_max_state.append(Q.max(axis=1))

    # DUMP DUMP DUMP
    np.savetxt(f"./data/Q_lambda_{_lambda}.csv", Q_max_state, delimiter=",")
        

    return Q

def write_video():
    image_width = 12
    image_height = 4
    lambdas = [0.0, 0.3, 0.5, 0.7, 0.9]
    
    df = np.array([np.genfromtxt(f"./data/Q_lambda_{_lambda}.csv", delimiter=",") for _lambda in lambdas])
    num_lambdas, num_episodes, num_states = df.shape
    
    fig = plt.figure(figsize=(25, 5))
    
    # Get Dimensions
    for i in range(num_lambdas):
        img = df[i, 0].reshape(image_height, -1)
        fig.add_subplot(1, num_lambdas, i+1)
        plt.axis("off")
        plt.title(f"λ = {lambdas[i]}")
        plt.imshow(img)
    fig.canvas.draw()
    image = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    fig.clear()
    
    height, width, layers = image.shape
    video = cv2.VideoWriter("./videos/td.avi", 0, 5, (width,height))
    
    for j in trange(num_episodes, desc="video"):
        for i in range(num_lambdas):
            img = df[i, j].reshape(image_height, -1)
            fig.add_subplot(1, num_lambdas, i+1)
            plt.axis("off")
            plt.title(f"λ = {lambdas[i]}")
            plt.imshow(img)
        fig.canvas.draw()
        video.write(cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR))
        fig.clear()

    cv2.destroyAllWindows()
    video.release()


def main():
    parser = argparse.ArgumentParser(description='Temporal Difference Learning Video.')
    parser.add_argument('--steps', action=argparse.BooleanOptionalAction)
    parser.add_argument('--video', action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    
        
    args_vars = vars(args)
    
    if (args_vars['steps']):
        env = gym.make('CliffWalking-v0')

        lambdas = [0.0, 0.3, 0.5, 0.7, 0.9]
        episodes = 500

        for _lambda in lambdas:
            sarsa(env, _lambda, episodes)

        env.close()
        
    if (args_vars['video']):
        write_video()

if __name__ == '__main__':
    main()
