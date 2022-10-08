import numpy as np
import torch


class ReplayBuffer:
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self, env, size, device):
        self._maxsize = size
        self._mem_counter = 0
        self._state_shape = env.observation_space.shape
        self._num_actions = env.action_space.n
        self._device = device

        self.states = torch.zeros(self._maxsize,
                                  *self._state_shape,
                                  dtype=torch.float32).to(self._device)
        self.actions = torch.zeros(self._maxsize,
                                   dtype=torch.int64).to(self._device)
        self.rewards = torch.zeros(self._maxsize,
                                   dtype=torch.float32).to(self._device)
        self.next_states = torch.zeros(self._maxsize,
                                       *self._state_shape,
                                       dtype=torch.float32).to(self._device)
        self.dones = torch.zeros(self._maxsize,
                                 dtype=torch.float32).to(self._device)

    def __call__(self, indices):
        return self.states[indices],\
            self.actions[indices],\
            self.rewards[indices],\
            self.next_states[indices],\
            self.dones[indices]

    def state_dict(self):
        dict = {
            'maxsize': self._maxsize,
            'mem_counter': self._mem_counter,
            'state_shape': self._state_shape,
            'num_actions': self._num_actions,
            'states': self.states.detach(),
            'actions': self.actions.detach(),
            'rewards': self.rewards.detach(),
            'next_state': self.next_states.detach(),
            'dones': self.dones.detach(),
        }

        return dict

    def load_state_dict(self, state_dict):
        self._maxsize = state_dict['maxsize']
        self._mem_counter = state_dict['mem_counter']
        self._state_shape = state_dict['state_shape']
        self._num_actions = state_dict['num_actions']

        self.states = torch.load(state_dict['states'],
                                 map_location=self._device)
        self.actions = torch.load(state_dict['actions'],
                                  map_location=self._device)
        self.rewards = torch.load(state_dict['rewards'],
                                  map_location=self._device)
        self.next_states = torch.load(state_dict['next_states'],
                                      map_location=self._device)
        self.dones = torch.load(state_dict['dones'],
                                map_location=self._device)

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer. Old transitions will be overwritten if the buffer is full.
        :param state: the agent's initial state
        :param action: the action taken by the agent
        :param reward: the reward the agent received
        :param next_state: the subsequent state
        :param done: whether the episode terminated
        """
        next_index = self._mem_counter % self._maxsize

        self.states[next_index] = torch.from_numpy(state)
        self.actions[next_index] = action
        self.rewards[next_index] = reward
        self.next_states[next_index] = torch.from_numpy(next_state)
        self.dones[next_index] = float(done)

        self._mem_counter += 1

    def last_frames(self, n):
        """
        Get last n frames in the memory buffer
        :param n: the number of frames
        :return: a batch of n frames
        """

        memory = min(self._mem_counter, self._maxsize) - 1
        assert self._mem_counter >= n, f"memory={memory} must have at least n={n} frames"

        index = self._mem_counter % self._maxsize
        indices = range(index-n, index)

        return self(indices)

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.
        :param batch_size: the number of transitions to sample
        :return: a mini-batch of sampled transitions
        """
        memory = min(self._mem_counter, self._maxsize) - 1
        assert self._mem_counter >= batch_size, f"memory={memory} must have at least batch_size={batch_size} frames"

        indices = np.random.randint(0, memory, size=batch_size)

        return self(indices)
