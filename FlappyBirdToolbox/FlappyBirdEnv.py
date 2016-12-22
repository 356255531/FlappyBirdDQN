from itertools import cycle
import random
import sys

import pygame
from pygame.locals import *


class Env(object):
    """docstring for Env"""

    def __init__(self, arg):
        super(Env, self).__init__()
        FPS = 30
        SCREENWIDTH = 288
        SCREENHEIGHT = 512
        # amount by which base can maximum shift to left
        PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
        BASEY = SCREENHEIGHT * 0.79
        # image and hitmask  dicts
        IMAGES, HITMASKS = {}, {}

        # list of all possible players (tuple of 3 positions of flap)
        PLAYERS = (
            'assets/sprites/redbird-upflap.png',
            'assets/sprites/redbird-midflap.png',
            'assets/sprites/redbird-downflap.png'
        )

        # list of backgrounds
        BACKGROUNDS = 'assets/sprites/background-day.png'

        # list of pipes
        PIPES = 'assets/sprites/pipe-green.png'
