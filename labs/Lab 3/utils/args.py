import argparse


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
