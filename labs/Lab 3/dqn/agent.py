from gym import spaces
import numpy as np
import torch

from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        # TODO: Initialise agent's networks, optimiser and replay buffer
        self.policy_network = DQN(observation_space=observation_space, action_space=action_space).to(device)
        self.target_network = DQN(observation_space=observation_space, action_space=action_space).to(device)
        
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        
        
        self.replay_buffer = replay_buffer
        

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        # TODO
        #   Optimise the TD-error over a single minibatch of transitions
        #   Sample the minibatch from the replay-memory
        #   using done (as a float) instead of if statement
        #   return loss

        raise NotImplementedError

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        # TODO update target_network parameters with policy_network parameters
        raise NotImplementedError

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        x = torch.zeros((1, 4, 84, 84), dtype=torch.float32)
        
        # Last three frames
        x[:,:-1,:,:] = torch.tensor(
            self.replay_buffer._encode_sample([-3, -2, -1])[0][:,0,:,:])
        # Last frame (state)
        x[:,-1,:,:] = torch.tensor(state)
        
        # Normalize
        x = x / 255
        
        # Q-values
        q_values = self.policy_network(x).detach()
        
        # Greedy action (max)
        action  = np.argmax(q_values)
        return action.item()
