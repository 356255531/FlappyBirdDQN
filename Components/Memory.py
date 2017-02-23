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
        self.frame_sets_size = frame_sets_size

        self.memory = deque()
        self.size = 0

    def add(self, element):
        if not self.__if_element_legal(element):
            print 'Add failed'
            return

        if self.size >= self.memory_limit:
            self.memory.popleft()
            self.size -= 1

        self.memory.append(element)
        self.size += 1

    def sample(self, size):
        if size <= 0:
            raise ValueError("sample return empty list")
            return []

        if size >= self.size:
            return list(self.memory)

        return rd.sample(self.memory, size)

    def __if_element_legal(self, element):
        return True


if __name__ == '__main__':
    import numpy as np
    a = np.empty((2, 3, 5))
    D = Memory(2, 3)
    D.add(a)
    D.add(a)
    D.add(a)
    print D.sample(1)
    print len(D.memory)
