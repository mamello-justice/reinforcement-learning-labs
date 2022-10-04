import argparse
import random
import numpy as np
import torch
import gym

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *

try:
    from tqdm import trange
except Exception:
    trange = range


def setup_args(default_device):
    parser = argparse.ArgumentParser(
        description=f'Deep Q-Learning (DQN) @see https://arxiv.org/abs/1312.5602')
    parser.add_argument('--device',
                        type=str,
                        default=default_device,
                        help='device to use for running models (default: %(default)s)')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='seed to be used on environment')
    parser.add_argument('--env',
                        type=str,
                        default="PongNoFrameskip-v4",
                        help="name of the atari game/env")
    parser.add_argument('--rbs',
                        type=int,
                        default=int(5e3),
                        help='size of replay buffer to be used by agent')
    parser.add_argument('--l-rate',
                        type=int,
                        default=int(1e-4),
                        help='learning rate for Adam optimizer')
    parser.add_argument('--discount',
                        type=int,
                        default=0.99,
                        help='discount factor for accumulated rewards')
    parser.add_argument('--steps',
                        type=int,
                        default=int(1e6),
                        help='total number of steps for which to run the environment')
    parser.add_argument('--batch',
                        type=int,
                        default=256,
                        help='number of transitions to optimize at the same time')
    parser.add_argument('--l-starts',
                        type=int,
                        default=10000,
                        help='number of steps before starting to learn')
    parser.add_argument('--l-freq',
                        type=int,
                        default=5,
                        help='number of iterations between every optimization step')
    parser.add_argument('--double-dqn',
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='whether to use double deep Q-learning (DDQN) @see https://arxiv.org/abs/1509.06461')
    parser.add_argument('--t-freq',
                        type=int,
                        default=1000,
                        help='number of iterations between every target network update')
    parser.add_argument('--eps-start',
                        type=int,
                        default=1.0,
                        help='e-greedy start threshold')
    parser.add_argument('--eps-end',
                        type=int,
                        default=0.01,
                        help='e-greedy end threshold')
    parser.add_argument('--eps-frac',
                        type=int,
                        default=0.1,
                        help='fraction of num-steps')
    parser.add_argument('--p-freq',
                        type=int,
                        default=10,
                        help='frequency at which to print info/stats')
    parser.add_argument('--out',
                        type=str,
                        help='path to model file')
    parser.add_argument('--in',
                        type=str,
                        help='path to model file')
    parser.add_argument('--visualize',
                        default=False,
                        action=argparse.BooleanOptionalAction,
                        help='whether to render for human mode or not')

    args = parser.parse_args()
    return vars(args)


def args_to_hyper_params(args):
    return {
        "device": args['device'],
        "seed": args['seed'],
        "env": args['env'],
        "replay-buffer-size": args['rbs'],
        "learning-rate": args['l_rate'],
        "discount-factor": args['discount'],
        "num-steps": args['steps'],
        "batch-size": args['batch'],
        "learning-starts": args['l_starts'],
        "learning-freq": args['l_freq'],
        "use-double-dqn": args['double_dqn'],
        "target-update-freq": args['t_freq'],
        "eps-start": args['eps_start'],
        "eps-end": args['eps_end'],
        "eps-fraction": args['eps_frac'],
        "print-freq": args['p_freq'],
    }


def init_env(hyper_params, args_vars):
    assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    if args_vars['visualize']:
        env = gym.make(hyper_params["env"], render_mode='human')
    else:
        env = gym.make(hyper_params["env"])
    env.seed(hyper_params["seed"])

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = ClipRewardEnv(env)

    # Preprocess the frames to 84 x 84 x 4
    env = WarpFrame(env)

    # Convert to torch type (Tensor) and shape (channel x height x width)
    env = PyTorchFrame(env)

    print(f"states = {env.observation_space.shape}")
    print(f"actions = {env.action_space.n}")
    return env


def train(env, agent, replay_buffer, hyper_params):
    eps_timesteps = hyper_params["eps-fraction"] * \
        float(hyper_params["num-steps"])
    episode_rewards = [0.0]

    state, _ = env.reset()
    for t in trange(hyper_params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (
            hyper_params["eps-end"] - hyper_params["eps-start"]
        )
        #  select random action if sample is less equal than eps_threshold
        if random.random() <= eps_threshold:
            action = env.action_space.sample()
        else:
            action = agent.act(torch.tensor(state))

        # take step in env
        next_state, reward, done, _, _ = env.step(action)

        # add state, action, reward, next_state, float(done) to reply memory - cast done to float
        replay_buffer.add(state=state,
                          action=action,
                          reward=reward,
                          next_state=next_state,
                          done=float(done))

        # add reward to episode_reward
        episode_rewards[-1] += reward

        state = next_state

        if done:
            state, _ = env.reset()
            episode_rewards.append(0.0)

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["learning-freq"] == 0
        ):
            agent.optimise_td_loss()

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["target-update-freq"] == 0
        ):
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if (
            done
            and hyper_params["print-freq"] is not None
            and len(episode_rewards) % hyper_params["print-freq"] == 0
        ):
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")


def main(hyper_params, args_vars):
    device = torch.device(hyper_params["device"])

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    env = init_env(hyper_params, args_vars)

    replay_buffer = ReplayBuffer(
        hyper_params["replay-buffer-size"], device=device)

    agent = DQNAgent(observation_space=env.observation_space,
                     action_space=env.action_space,
                     replay_buffer=replay_buffer,
                     use_double_dqn=hyper_params['use-double-dqn'],
                     lr=hyper_params['learning-rate'],
                     batch_size=hyper_params['batch-size'],
                     gamma=hyper_params['discount-factor'],
                     device=device)

    in_file = args_vars['in']
    if in_file is not None:
        try:
            agent.load(in_file)
        except FileNotFoundError:
            pass

    try:
        train(env, agent, replay_buffer, hyper_params)
    finally:
        out_file = args_vars['out']
        if out_file is not None:
            agent.save(out_file)


if __name__ == "__main__":
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    args_vars = setup_args(default_device=default_device)
    hyper_params = args_to_hyper_params(args_vars)
    main(hyper_params=hyper_params, args_vars=args_vars)
