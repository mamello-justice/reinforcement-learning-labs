import argparse
import os
import numpy as np
import pandas as pd
import gym
import matplotlib.pyplot as plt
from collections import namedtuple
from datetime import datetime
from time import sleep

from value_function import ValueFunction

try:
    # NB: Install tqdm by `pip install tqdm` to visualize the progress of the training
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

# Reinforcement Learning (COMS4061A)
# Function Approximation
# Mamello:1851317


EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])

data_dir = "data"
videos_dir = "videos"
mountain_car_env = 'MountainCar-v0'


def now_formatted():
    return datetime.now().__str__().replace(" ", "_").replace(":", "_")


def semi_g_sarsa_control_epsilon_greedy(env, num_episodes, epsilon=0.1, alpha=0.1):
    # Initialize stats
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Reset environment
    env.reset()

    # Initialize Q value function for tiles
    Q = ValueFunction(alpha=alpha, n_actions=env.action_space.n)

    for i_episode in trange(num_episodes):
        # Initialize
        observation, reward, done = env.reset()[0], 0, False
        action = Q.act(observation, epsilon=epsilon)

        while not done:
            # Take action and observe next observation and reward
            next_observation, reward, done, _, _ = env.step(action)

            # Update statistics after getting a reward - use within loop, call the following lines
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] += 1

            if done:
                Q.update(reward, observation, action)
                continue

            # Choose next action using policy
            next_action = Q.act(next_observation, epsilon=epsilon)

            # are we missing discounting?
            target = reward + Q(next_observation, next_action)
            Q.update(target, observation, action)

            observation = next_observation
            action = next_action

    return Q, stats


def plot_graph(num_runs, num_episodes):
    env = gym.make(mountain_car_env)
    sum_episode_lengths = np.zeros(num_episodes)

    for run in range(num_runs):
        _, stats = semi_g_sarsa_control_epsilon_greedy(
            env, num_episodes=num_episodes)

        sum_episode_lengths += stats.episode_lengths

    num_steps_average = sum_episode_lengths / num_runs

    plt.plot(range(num_episodes), num_steps_average)
    plt.title(
        f"Number of steps per episode ({mountain_car_env}) averaged over {num_runs} run(s)")
    plt.ylabel("Number of steps")
    plt.xlabel("Episode #")
    plt.yscale('log')
    plt.show()

    env.close()


def generate_video(num_episodes):
    env = gym.make(mountain_car_env)
    Q, _ = semi_g_sarsa_control_epsilon_greedy(
        env, num_episodes=num_episodes)

    # Close training env
    env.close()

    # Rendering environment (slower due to render_mode)
    rgb_env = gym.make(mountain_car_env, render_mode="rgb_array_list")
    video_env = gym.wrappers.RecordVideo(
        rgb_env, videos_dir, name_prefix=f"mountain-car-{now_formatted()}")
    state, _ = video_env.reset()

    done = False
    while not done:
        action = Q.act(state)
        state, _, done, _, _ = video_env.step(action)

        if done:
            video_env.reset()

    # Close video env
    video_env.close()

    # Close parent env
    rgb_env.close()


def save_weights(Q):
    weights_filename = f"{data_dir}/value_function_weights_{now_formatted()}.txt"

    np.savetxt(weights_filename, Q.weights)


def save_stats(stats):
    stats_filename = f"{data_dir}/stats_{now_formatted()}.csv"

    df = pd.DataFrame(stats._asdict())
    df.to_csv(stats_filename)


def main():
    parser = argparse.ArgumentParser(
        description=f'Value Function Approximation ({mountain_car_env})')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction)
    parser.add_argument('--graph', action=argparse.BooleanOptionalAction)
    parser.add_argument('--video', action=argparse.BooleanOptionalAction)
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=500)

    print(parser.description)

    args = parser.parse_args()
    args_vars = vars(args)

    env = gym.make(mountain_car_env)

    # Visualize environment
    print("observation space\n", env.observation_space)
    print()
    print("action space\n", env.action_space)
    print()

    # observation = tuple(position, velocity)
    observation, _ = env.reset()
    print("example_observation = ", observation)
    print()

    num_runs = args_vars['runs']
    num_episodes = args_vars['episodes']
    save = args_vars['save']
    graph = args_vars['graph']
    video = args_vars['video']

    # Task 1
    if graph or not (video or save):
        plot_graph(num_runs, num_episodes)

    # Task 2
    if video or not (graph or save):
        assert os.path.isdir(videos_dir)
        generate_video(num_episodes)

    # Optional (Save weights and stats)
    if save:
        assert os.path.isdir(data_dir)
        Q, stats = semi_g_sarsa_control_epsilon_greedy(
            env, num_episodes=num_episodes)
        save_weights(Q)
        save_stats(stats)

    env.close()


if __name__ == "__main__":
    main()
