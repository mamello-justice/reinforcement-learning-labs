import argparse
import random

import numpy as np
import pandas as pd
import torch

import gym
from gym import wrappers

from dqn.agent import DQNAgent
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
    parser.add_argument('--memory',
                        type=int,
                        default=int(5e3),
                        help='size of replay buffer to be used by agent')
    parser.add_argument('--lr',
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
                        default=int(1e4),
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
                        default=int(1e3),
                        help='number of iterations between every target network update')
    parser.add_argument('--eps-start',
                        type=int,
                        default=1.0,
                        help='e-greedy start threshold')
    parser.add_argument('--eps-end',
                        type=int,
                        default=0.01,
                        help='e-greedy end threshold')
    parser.add_argument('--eps-fraction',
                        type=int,
                        default=0.1,
                        help='fraction of num-steps')
    parser.add_argument('--stat-freq',
                        type=int,
                        default=10,
                        help='frequency at which to log stats')
    parser.add_argument('--log-weights',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='log network weights (NB: this slows down learning significantly)')
    parser.add_argument('--log-dir',
                        type=str,
                        help='path at which to log stats')
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
        "seed": args.get('seed'),
        "env": args.get('env'),
        "use-double-dqn": args.get('double_dqn'),
        "replay-buffer-size": args.get('memory'),
        "batch-size": args.get('batch'),
        "num-steps": args.get('steps'),
        "learning-starts": args.get('l_starts'),
        "learning-freq": args.get('l_freq'),
        "target-update-freq": args.get('t_freq'),
        "learning-rate": args.get('lr'),
        "discount-factor": args.get('discount'),
        "eps-start": args.get('eps_start'),
        "eps-end": args.get('eps_end'),
        "eps-fraction": args.get('eps_fraction')
    }


class TrainingEnvironment:
    def __init__(self, env_name, seed, steps):
        assert "NoFrameskip" in env_name, "Require environment with no frameskip"

        self.seed = seed
        self.steps = steps

        self.env = gym.make(env_name, render_mode="rgb_array")
        self.env.seed(seed)

        self.env = wrappers.RecordVideo(self.env, "videos")
        self.env = NoopResetEnv(self.env, noop_max=30)
        self.env = MaxAndSkipEnv(self.env, skip=4)
        self.env = EpisodicLifeEnv(self.env)
        self.env = FireResetEnv(self.env)
        self.env = ClipRewardEnv(self.env)
        self.env = WarpFrame(self.env)
        self.env = PyTorchFrame(self.env)

    def unwrap(self):
        return self.env

    def trigger(self, x):
        if x % 100 == 0:
            print(x)
        return x > self.steps - 600

    def __str__(self):
        return f"seed = {self.seed}\nstates = {self.env.observation_space.shape}\nactions = {self.env.action_space.n}"


def get_epsilon_threshold(t, hyper_params):
    diff = hyper_params['eps-end'] - hyper_params['eps-start']
    fraction = float(t) / hyper_params['eps-fraction'] * \
        float(hyper_params['num-steps'])
    fraction = min(1.0, fraction)
    return hyper_params['eps-start'] + fraction * diff


if __name__ == "__main__":
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    args_vars = setup_args(default_device=default_device)
    hyper_params = args_to_hyper_params(args_vars)

    device = torch.device(args_vars.get("device"))
    print("device = %s" % device)

    np.random.seed(hyper_params['seed'])
    random.seed(hyper_params['seed'])

    train_env = TrainingEnvironment(env_name=hyper_params['env'],
                                    seed=hyper_params['seed'],
                                    steps=hyper_params['num-steps'])

    env = train_env.unwrap()

    agent = DQNAgent(env=env,
                     memory_size=hyper_params['replay-buffer-size'],
                     use_double_dqn=hyper_params['use-double-dqn'],
                     lr=hyper_params['learning-rate'],
                     batch_size=hyper_params['batch-size'],
                     gamma=hyper_params['discount-factor'],
                     device=device,
                     log_dir=args_vars['log_dir'],
                     log_weights=args_vars['log_weights'])

    in_file = args_vars.get('in')
    if in_file is not None:
        try:
            agent.load(in_file)
        except FileNotFoundError:
            print("model file not found: %s" % in_file)

    try:
        state, _ = env.reset()
        for t in trange(hyper_params['num-steps']):
            epsilon_threshold = get_epsilon_threshold(t, hyper_params)

            #  select random action if sample is less equal than eps_threshold
            if (t <= hyper_params['learning-starts'] or random.random() <= epsilon_threshold):
                action = env.action_space.sample()
            else:
                action = agent.act(torch.tensor(state))

            # take step in env
            next_state, reward, done, _, _ = env.step(action)

            # add state, action, reward, next_state, float(done) to reply memory - cast done to float
            agent.remember(state=state,
                           action=action,
                           reward=reward,
                           next_state=next_state,
                           done=float(done))

            # Update state
            state = next_state

            if done:
                state, _ = env.reset()

                if (agent.num_episodes() % args_vars.get('stat_freq') == 0):
                    explore_time = int(100 * epsilon_threshold)
                    agent.log(t)
                    agent.tb_w.add_scalar("Explore time", explore_time, t)

            if t > hyper_params['learning-starts']:
                if t % hyper_params['learning-freq'] == 0:
                    agent.optimise_td_loss()

                if t % hyper_params['target-update-freq'] == 0:
                    agent.update_target_network()

    finally:
        agent.tb_w.close()
        out_file = args_vars.get('out')
        if out_file is not None:
            agent.save(out_file)
