from datetime import datetime
from gym import spaces
import torch
import torch.nn as nn

from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer



class DQNAgent:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma,
        device,
    ):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        # Q-networks
        self.Q = DQN(observation_space=observation_space, action_space=action_space).to(self.device)
        self.Q_target = DQN(observation_space=observation_space, action_space=action_space).to(self.device)
        
        # Loss
        self.loss = nn.MSELoss()
        
        # Adam optimizer
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
        
        # Replay buffer/memory
        self.replay_buffer = replay_buffer
        

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        with torch.no_grad(): # Not used in gradient calculation
            values_target = self.Q_target(next_states).max(dim=1, keepdim=True)[0].flatten()
            target = rewards + self.gamma * values_target * (1 - dones)   
        
        prediction = self.Q(states).gather(dim=1, index=actions.reshape((-1, 1))).flatten()
        
        loss = self.loss(prediction, target).to(self.device)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        for target, policy in zip(self.Q_target.parameters(), self.Q.parameters()):
            target.data.copy_(policy.data)

    def act(self, state: torch.Tensor):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        # Last four frames (states)
        x, _, _, _, _ = self.replay_buffer._encode_sample([-4, -3, -2, -1])
        
        # Replace first frame (state)
        x[-1,:,:,:] = state
        
        # Normalize
        x = x / 255
        
        # Q-values
        q_values = self.Q(x).detach()
        
        # Greedy action (max)
        action  = torch.argmax(q_values, dim=1)
        
        return action[-1]
    
    def save(self, path):
        print('saving models...')
        
        checkpoint = {
            'Q': self.Q.state_dict(),
            'Q_target': self.Q_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': self.loss,
            'replay_buffer': self.replay_buffer._storage
        }
        
        torch.save(checkpoint, path)

    
    def load(self, path):
        print('loading models...')
        
        checkpoint = torch.load(path)
        
        self.Q.load_state_dict(checkpoint.get('Q'), map_location=self.device)
        self.Q.train()
        
        self.Q_target.load_state_dict(checkpoint.get('Q_target'), map_location=self.device)
        self.Q_target.train()
        
        self.optimizer.load_state_dict(checkpoint.get('optimizer'))
        
        self.loss = checkpoint.get('loss')
        
        self.replay_buffer._storage = checkpoint.get('replay_buffer')
        self.replay_buffer._next_idx = len(self.replay_buffer._storage) % self.replay_buffer._maxsize