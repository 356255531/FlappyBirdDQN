import random as rd
from collections import deque

__author__ = 'Zhiwei'


class Memory(object):
    """
        A memory zone to save experience
    """

    def __init__(
            self,
            memory_limit,
            frame_sets_size
    ):
        super(Memory, self).__init__()
        self.memory_limit = memory_limit

        self.memory = deque()
        self.size = 0

    def add(self, element):
        if self.size >= self.memory_limit:
            self.memory.popleft()

        self.memory.append(element)

    def sample(self, size):
        if size <= 0:
            raise ValueError("sample return empty list")
            return []

        if size >= self.size:
            return list(self.memory)

        return rd.sample(self.memory, size)


if __name__ == '__main__':
    import numpy as np
    a = np.empty((2, 3, 5))
    D = Memory(2, 3)
    D.add(a)
    D.add(a)
    D.add(a)
    print D.sample(1)
    print len(D.memory)
