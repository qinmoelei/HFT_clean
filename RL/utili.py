import torch
import numpy as np
from collections import deque, namedtuple
import random


# traiditional 1 step td error
class ReplayMemory():

    def __init__(self, memory_size, state_shape, info_len):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.state_shape = state_shape
        self.memory_counter = 0
        self.memory_size = memory_size
        self.state_memory = torch.FloatTensor(self.memory_size, state_shape)
        self.action_memory = torch.LongTensor(self.memory_size)
        self.reward_memory = torch.FloatTensor(self.memory_size)
        self.done_memory = torch.FloatTensor(self.memory_size)
        self.state__memory = torch.FloatTensor(self.memory_size, state_shape)

    def reset(self):
        self.memory_counter = 0
        self.state_memory = torch.FloatTensor(self.memory_size,
                                              self.state_shape)
        self.action_memory = torch.LongTensor(self.memory_size)
        self.reward_memory = torch.FloatTensor(self.memory_size)
        self.done_memory = torch.FloatTensor(self.memory_size)
        self.state__memory = torch.FloatTensor(self.memory_size,
                                               self.state_shape)

    def store(self, s, a, r, s_, done):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = s
        self.action_memory[index] = torch.LongTensor([a.tolist()])
        self.reward_memory[index] = torch.FloatTensor([r])
        self.state__memory[index] = s_
        self.done_memory[index] = torch.FloatTensor([done])

        self.memory_counter += 1

    def sample(self, size):
        sample_index = np.random.choice(self.memory_size, size)
        state_sample = torch.FloatTensor(size,
                                         self.state_shape).to(self.device)
        action_sample = torch.LongTensor(size, 1).to(self.device)
        reward_sample = torch.FloatTensor(size, 1).to(self.device)
        state__sample = torch.FloatTensor(size,
                                          self.state_shape).to(self.device)
        for index in range(sample_index.size):
            state_sample[index] = self.state_memory[sample_index[index]]
            action_sample[index] = self.action_memory[sample_index[index]]
            reward_sample[index] = self.reward_memory[sample_index[index]]
            state__sample[index] = self.state__memory[sample_index[index]]
        return state_sample, action_sample, reward_sample, state__sample


# multi step replay buffer: set parallel_env=1 and n_step=3600 in our case for defalt configuration
class Multi_step_ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self,
                 buffer_size,
                 batch_size,
                 device,
                 seed,
                 gamma,
                 n_step,
                 parallel_env=1):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.parallel_env = parallel_env
        self.n_step_buffer = [
            deque(maxlen=self.n_step) for i in range(parallel_env)
        ]
        self.iter_ = 0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # if we want to have multi core
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        self.n_step_buffer[self.iter_].append(
            (state, action, reward, next_state, done))
        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(
                self.n_step_buffer[self.iter_])
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)
        self.iter_ += 1

    def reset(self):
        self.memory = deque(maxlen=self.buffer_size)

    def calc_multistep_return(self, n_step_buffer):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * n_step_buffer[idx][2]

        return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[
            -1][3], n_step_buffer[-1][4]

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.stack([e.state for e in experiences
                      if e is not None])).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences
                       if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences
                       if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(
            np.stack([e.next_state for e in experiences
                      if e is not None])).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None
                       ]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# multi step replay buffer: set parallel_env=1 and n_step=3600 in our case for defalt configuration
class Multi_step_ReplayBuffer_info:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self,
                 buffer_size,
                 batch_size,
                 device,
                 seed,
                 gamma,
                 n_step,
                 parallel_env=1):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=[
                                         "state", "action", "reward",
                                         "next_state", "done", "info"
                                     ])
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.parallel_env = parallel_env
        self.n_step_buffer = [
            deque(maxlen=self.n_step) for i in range(parallel_env)
        ]
        self.iter_ = 0

    def add(self, state, action, reward, next_state, done, info: dict):
        """Add a new experience to memory."""
        # if we want to have multi core
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        self.n_step_buffer[self.iter_].append(
            (state, action, reward, next_state, done, info))
        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            state, action, reward, next_state, done, info = self.calc_multistep_return(
                self.n_step_buffer[self.iter_])
            e = self.experience(state, action, reward, next_state, done, info)
            self.memory.append(e)
        self.iter_ += 1

    def reset(self):
        self.memory = deque(maxlen=self.buffer_size)

    def calc_multistep_return(self, n_step_buffer):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * n_step_buffer[idx][2]
        self.info_key = n_step_buffer[0][5].keys()

        return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[
            -1][3], n_step_buffer[-1][4], n_step_buffer[0][5]

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.stack([e.state for e in experiences
                      if e is not None])).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences
                       if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences
                       if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(
            np.stack([e.next_state for e in experiences
                      if e is not None])).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None
                       ]).astype(np.uint8)).float().to(self.device)
        infos = dict()
        for key in self.info_key:
            infos[key] = torch.from_numpy(
                np.stack([e.info[key] for e in experiences if e is not None
                          ]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones, infos)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


if __name__ == "__main__":
    pass