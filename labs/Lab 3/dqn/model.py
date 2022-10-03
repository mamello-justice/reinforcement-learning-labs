from gym import spaces
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        assert (
            type(observation_space) == spaces.Box
        ), "observation_space must be of type Box"
        assert (
            len(observation_space.shape) == 3
        ), "observation space must have the form channels x width x height"
        assert (
            type(action_space) == spaces.Discrete
        ), "action_space must be of type Discrete"
        
        # 1 channel -> 16 features -> 32 features -> 32*9*9=2592 features -> 256 -> # actions
        self.model = nn.Sequential(nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=0),
                                    nn.ReLU(),
                                    nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(32*9*9, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, action_space.n))

    def forward(self, x):
        return self.model(x)
