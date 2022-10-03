import random
import numpy as np
import torch
import gym

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
from utils.args import setup_args

try:
    from tqdm import trange
except Exception:
    trange = range


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


def main(hyper_params):
    device = torch.device(hyper_params["device"])

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
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

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"], device=device)

    agent = DQNAgent(observation_space=env.observation_space,
                     action_space=env.action_space,
                     replay_buffer=replay_buffer,
                     use_double_dqn=hyper_params['use-double-dqn'],
                     lr=hyper_params['learning-rate'],
                     batch_size=hyper_params['batch-size'],
                     gamma=hyper_params['discount-factor'],
                     device=device)

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


if __name__ == "__main__":
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    args_vars = setup_args(default_device=default_device)
    hyper_params = args_to_hyper_params(args_vars)
    main(hyper_params=hyper_params)
