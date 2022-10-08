import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, env, memory_size, use_double_dqn, lr, batch_size, gamma, device, log_dir):
        self.num_actions = env.action_space.n
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.use_double_dqn = use_double_dqn
        self.episode_rewards = [0.0]

        # Tensorboard
        self.tb_w = tb.SummaryWriter(log_dir)

        # Q-networks
        self.Q = self._build_model().to(self.device)
        self.Q_target = self._build_model().to(self.device)

        # Loss
        self.loss = nn.MSELoss()

        # Adam optimizer
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)

        # Replay buffer/memory
        self.memory = ReplayBuffer(env=env, size=memory_size, device=device)

        # Indexing
        self.idx = 0

    def _build_model(self):
        # 1 channel -> 16 features -> 32 features -> 32*9*9=2592 features -> 256 -> # actions
        return nn.Sequential(nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=0),
                             nn.ReLU(),
                             nn.Conv2d(16, 32, kernel_size=4,
                                       stride=2, padding=0),
                             nn.ReLU(),
                             nn.Flatten(),
                             nn.Linear(32*9*9, 256), nn.ReLU(),
                             nn.Linear(256, self.num_actions))

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        # Normalize memory states
        states = states / 255
        next_states = next_states / 255

        # Targets
        with torch.no_grad():  # Not used in gradient calculation
            if self.use_double_dqn:
                target_actions = target_values = self.Q(next_states)\
                    .argmax(dim=1, keepdim=True)

                target_values = self.Q_target(next_states)\
                    .gather(dim=1, index=target_actions).flatten()

            else:
                # torch.max() -> Tensor[values], Tensor[indices]
                target_values = self.Q_target(next_states)\
                    .max(dim=1, keepdim=True)[0].flatten()

            targets = rewards + \
                self.gamma * target_values * (1 - dones)

        # Online
        values = self.Q(states)\
            .gather(dim=1, index=actions.unsqueeze(-1))\
            .flatten()

        # Loss
        loss = self.loss(values, targets).to(self.device)

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log to tensorboard
        self.tb_w.add_scalar("Loss", loss.item(), self.idx)
        for name, weight in self.Q.named_parameters():
            self.tb_w.add_histogram('Q-%s' % name, weight, self.idx)
            self.tb_w.add_histogram('Q-%s.grad' % name, weight, self.idx)

        for name, weight in self.Q_target.named_parameters():
            self.tb_w.add_histogram('Q-target-%s' % name, weight, self.idx)
            self.tb_w.add_histogram('Q-target-%s' % name, weight, self.idx)

        self.idx += 1

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.Q_target.load_state_dict(self.Q.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        # Rewards
        self.episode_rewards[-1] += reward

        if done:
            self.episode_rewards.append(0.0)

    def num_episodes(self):
        return len(self.episode_rewards)

    def act(self, state: torch.Tensor):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        # Last four frames (states)
        batch, _, _, _, _ = self.memory.last_frames(4)

        # Replace first frame (state)
        batch[0, :, :, :] = state

        # Normalize memory states
        batch = batch / 255

        with torch.no_grad():
            # Greedy action (max)
            return self.Q(batch).argmax(dim=1, keepdim=True)[1].view(1, 1)

    def log(self, t):
        self.tb_w.add_scalar("Episodes", self.num_episodes(), t)

        name = "Mean Reward (100 episodes)"
        mean_100ep_reward = round(np.mean(self.episode_rewards[-101:-1]), 1)
        self.tb_w.add_scalar(name, mean_100ep_reward, t)

    def save(self, path):
        print('saving models...')

        checkpoint = {
            'Q': self.Q.state_dict(),
            'Q_target': self.Q_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': self.loss,
            'memory': self.memory.state_dict()
        }

        torch.save(checkpoint, path)

    def load(self, path):
        print('loading models...')

        checkpoint = torch.load(path)

        self.Q.load_state_dict(checkpoint.get('Q'), map_location=self.device)
        self.Q.train()

        self.Q_target.load_state_dict(
            checkpoint.get('Q_target'),
            map_location=self.device)
        self.Q_target.train()

        self.optimizer.load_state_dict(checkpoint.get('optimizer'))

        self.loss = checkpoint.get('loss')

        self.memory.load_state_dict(checkpoint.get('memory'))
