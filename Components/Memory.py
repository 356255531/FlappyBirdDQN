__author__ = 'Zhiwei'
from copy import deepcopy
import random as rd


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

        self.memory = []
        self.size = 0

    def add(self, element):
        if not self.__if_element_legal(element):
            print 'Add failed'
            return
        if self.size >= self.memory_limit:
            self.memory.pop()
            self.size -= 1
        self.memory.append(element)
        self.size += 1

    def sample(self, size):
        if size == 0:
            return []
        if size > self.size:
            return deepcopy(self.memory)
        ret_batch = [deepcopy(self.memory[i]) for i in rd.sample(xrange(self.size), size)]
        return ret_batch

    def __if_element_legal(self, element):
        pass


if __name__ == '__main__':
    import numpy as np
    a = np.empty((2, 3, 5))
    print len(a.shape)
