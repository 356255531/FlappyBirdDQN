from itertools import cycle
import random
import sys

import pygame
from pygame.locals import *


class Env(object):
    """docstring for Env"""

    def __init__(self):
        super(Env, self).__init__()

        self.FPS = 30
        self.SCREENWIDTH = 288
        self.SCREENHEIGHT = 512
        # amount by which base can maximum shift to left
        self.PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
        self.BASEY = SCREENHEIGHT * 0.79
        # image and hitmask  dicts
        self.IMAGES, self.HITMASKS = {}, {}

        # list of all possible players (tuple of 3 positions of flap)
        self.PLAYERS = (
            'assets/sprites/redbird-upflap.png',
            'assets/sprites/redbird-midflap.png',
            'assets/sprites/redbird-downflap.png'
        )

        # list of backgrounds
        self.BACKGROUNDS = 'assets/sprites/background-day.png'

        # list of pipes
        self.PIPES = 'assets/sprites/pipe-green.png'

        # if game end
        self.done = False


        pygame.init()
        self.FPSCLOCK = pygame.time.Clock()
        self.SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
        pygame.display.set_caption('Flappy Bird')

        # numbers sprites for score display
        self.IMAGES['numbers'] = (
            pygame.image.load('assets/sprites/0.png').convert_alpha(),
            pygame.image.load('assets/sprites/1.png').convert_alpha(),
            pygame.image.load('assets/sprites/2.png').convert_alpha(),
            pygame.image.load('assets/sprites/3.png').convert_alpha(),
            pygame.image.load('assets/sprites/4.png').convert_alpha(),
            pygame.image.load('assets/sprites/5.png').convert_alpha(),
            pygame.image.load('assets/sprites/6.png').convert_alpha(),
            pygame.image.load('assets/sprites/7.png').convert_alpha(),
            pygame.image.load('assets/sprites/8.png').convert_alpha(),
            pygame.image.load('assets/sprites/9.png').convert_alpha()
        )

        # game over sprite
        self.IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
        # message sprite for welcome screen
        self.IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
        # base (ground) sprite
        self.IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

        self.IMAGES['background'] = pygame.image.load(BACKGROUNDS).convert()

        # select random player sprites
        self.IMAGES['player'] = (
            pygame.image.load(PLAYERS[0]).convert_alpha(),
            pygame.image.load(PLAYERS[1]).convert_alpha(),
            pygame.image.load(PLAYERS[2]).convert_alpha(),
        )

        # select random pipe sprites
        self.IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPES).convert_alpha(), 180),
            pygame.image.load(PIPES).convert_alpha(),
        )

        # hismask for pipes
        self.HITMASKS['pipe'] = (
            getHitmask(self.IMAGES['pipe'][0]),
            getHitmask(self.IMAGES['pipe'][1]),
        )

        # hitmask for player
        self.HITMASKS['player'] = (
            getHitmask(self.IMAGES['player'][0]),
            getHitmask(self.IMAGES['player'][1]),
            getHitmask(self.IMAGES['player'][2]),
        )

    def init_game(self):
        """Shows welcome screen animation of flappy bird"""
        # index of player to blit on screen
        playerIndexGen = cycle([0, 1, 2, 1])
        # iterator used to change playerIndex after every 5th iteration

        playery = int((self.SCREENHEIGHT - self.IMAGES['player'][0].get_height()) / 2)

        basex = 0

        # player shm for up-down motion on welcome screen
        playerShmVals = {'val': 0, 'dir': 1}

        self.movementInfo =  {
            'playery': playery + playerShmVals['val'],
            'basex': basex,
            'playerIndexGen': playerIndexGen,
        }

        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)

        baseShift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        # get 2 new pipes to add to upperPipes lowerPipes list
        self.newPipe1 = getRandomPipe()
        self.newPipe2 = getRandomPipe()

        # list of upper pipes
        self.upperPipes = [
            {'x': self.SCREENWIDTH + 200, 'y': self.newPipe1[0]['y']},
            {'x': self.SCREENWIDTH + 200 + (self.SCREENWIDTH / 2), 'y': self.newPipe2[0]['y']},
        ]

        # list of lowerpipe
        self.lowerPipes = [
            {'x': self.SCREENWIDTH + 200, 'y': self.newPipe1[1]['y']},
            {'x': self.SCREENWIDTH + 200 + (self.SCREENWIDTH / 2), 'y': self.newPipe2[1]['y']},
        ]

        self.pipeVelX = -4

        # player velocity, max velocity, downward accleration, accleration on flap
        self.playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   # min vel along Y, max ascend speed
        self.playerAccY    =   1   # players downward accleration
        self.playerFlapAcc =  -9   # players speed on flapping
        self.playerFlapped = False # True when player flaps

    def step(self, action):
        if action == 1:
            if self.playery > -2 * self.IMAGES['player'][0].get_height():
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        crashTest = self.checkCrash(
                                {
                                    'x': self.playerx, 
                                    'y': self.playery, 
                                    'index': self.playerIndex
                                },
                                self.upperPipes, 
                                self.lowerPipes
                    )
        if crashTest[0]:
            self.done = True
            return 0, self.done

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = self.movementInfo['playerIndexGen'].next()
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapY = random.randrange(0, int(self.BASEY * 0.6 - self.PIPEGAPSIZE))
        gapY += int(self.BASEY * 0.2)
        pipeHeight = self.IMAGES['pipe'][0].get_height()
        pipeX = self.SCREENWIDTH + 10

        return [
            {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
            {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
        ]

    def getHitmask(self, image):
        """returns a hitmask using an image's alpha."""
        mask = []
        for x in range(image.get_width()):
            mask.append([])
            for y in range(image.get_height()):
                mask[x].append(bool(image.get_at((x,y))[3]))
        return mask
