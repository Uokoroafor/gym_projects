import random
from collections import deque


class ReplayBuffer:
    def __init__(self, size):
        """Replay buffer initialisation

        Args:
            size(int): The max number of objects that can be stored by the replay buffer
        """
        self.size = size
        self.buffer = deque([], maxlen=size)

    def push(self, transition):
        """Adds a transition to the replay buffer

        Args:
            transition(obj): to be stored in replay buffer
        """
        self.buffer.append(transition)

    def sample(self, batch_size):
        """Randomly sample the replay buffer

        Args:
            batch_size: size of sample

        Returns:
            sampled list from buffer without replacement
        """
        return random.sample(self.buffer, batch_size)
