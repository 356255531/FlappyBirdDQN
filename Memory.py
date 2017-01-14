__author__ = 'Zhiwei'
import random as rd
from copy import deepcopy


class Memory(object):
    """docstring for Memory"""

    def __init__(self):
        super(Memory, self).__init__()
        self.memory = []
        self.size = 0

    def add(self, element):
        if self.size >= 1000000:
            self.memory.pop()
            self.size -= 1
        self.memory.append(element)
        self.size += 1

    def sample(self, size):
        if size > self.size:
            return deepcopy(self.memory)
        ret_batch = [deepcopy(self.memory[i]) for i in rd.sample(xrange(self.size), size)]
        return ret_batch


if __name__ == '__main__':
