from itertools import cycle
import random

import pygame
from pygame.locals import *


class FlappyBirdEnv(object):
    """
    The flappy bird simulator for DRL algorithm
    API:
        init_game(preprocess=False):
            Reset game to initial state (if perform learning)
            return: the initial state (n frames here n = 4)

        step()
    """

    def __init__(self, rel_path='', frame_set_size=4, mode='play'):
        super(FlappyBirdEnv, self).__init__()

        self.frame_set_size = frame_set_size

        if mode == 'play':
            self.FPS = 30
        else:
            self.FPS = 3000
        self.SCREENWIDTH = 288
        self.SCREENHEIGHT = 512
        # amount by which base can maximum shift to left
        self.PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
        self.BASEY = self.SCREENHEIGHT * 0.79
        # image and hitmask  dicts
        self.IMAGES, self.HITMASKS = {}, {}

        # list of all possible players (tuple of 3 positions of flap)
        self.PLAYERS = (
            rel_path + 'assets/sprites/redbird-upflap.png',
            rel_path + 'assets/sprites/redbird-midflap.png',
            rel_path + 'assets/sprites/redbird-downflap.png'
        )

        # list of backgrounds
        self.BACKGROUNDS = rel_path + 'assets/sprites/background-day.png'

        # list of pipes
        self.PIPES = rel_path + 'assets/sprites/pipe-green.png'

        pygame.init()
        self.FPSCLOCK = pygame.time.Clock()
        self.SCREEN = pygame.display.set_mode((self.SCREENWIDTH, self.SCREENHEIGHT))
        pygame.display.set_caption('Flappy Bird')

        # game over sprite
        self.IMAGES['gameover'] = pygame.image.load(rel_path + 'assets/sprites/gameover.png').convert_alpha()
        # message sprite for welcome screen
        self.IMAGES['message'] = pygame.image.load(rel_path + 'assets/sprites/message.png').convert_alpha()
        # base (ground) sprite
        self.IMAGES['base'] = pygame.image.load(rel_path + 'assets/sprites/base.png').convert_alpha()

        self.IMAGES['background'] = pygame.image.load(self.BACKGROUNDS).convert()

        # select random player sprites
        self.IMAGES['player'] = (
            pygame.image.load(self.PLAYERS[0]).convert_alpha(),
            pygame.image.load(self.PLAYERS[1]).convert_alpha(),
            pygame.image.load(self.PLAYERS[2]).convert_alpha(),
        )

        # select random pipe sprites
        self.IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(self.PIPES).convert_alpha(), 180),
            pygame.image.load(self.PIPES).convert_alpha(),
        )

        # hismask for pipes
        self.HITMASKS['pipe'] = (
            self.getHitmask(self.IMAGES['pipe'][0]),
            self.getHitmask(self.IMAGES['pipe'][1]),
        )

        # hitmask for player
        self.HITMASKS['player'] = (
            self.getHitmask(self.IMAGES['player'][0]),
            self.getHitmask(self.IMAGES['player'][1]),
            self.getHitmask(self.IMAGES['player'][2]),
        )

    def get_frame_set(self):
        if self.done:
            frame_set = [pygame.surfarray.array2d(
                pygame.display.get_surface()).T] * self.frame_set_size
            return frame_set

        frame_set = []
        for x in xrange(4):
            frame = pygame.surfarray.array2d(
                pygame.display.get_surface()).T
            frame_set.append(frame)
            self.one_iter(0)
        return frame_set

    def init_game(self):
        """
        Init the game and render the start frames
        return: initial frame set
        """
        self.done = False

        # index of player to blit on screen
        playerIndexGen = cycle([0, 1, 2, 1])
        # iterator used to change playerIndex after every 5th iteration

        playery = int((self.SCREENHEIGHT - self.IMAGES['player'][0].get_height()) / 2)

        basex = 0

        # player shm for up-down motion on welcome screen
        playerShmVals = {'val': 0, 'dir': 1}

        self.movementInfo = {
            'playery': playery + playerShmVals['val'],
            'basex': basex,
            'playerIndexGen': playerIndexGen,
        }

        self.score = self.playerIndex = self.loopIter = 0
        self.playerx, self.playery = int(self.SCREENWIDTH * 0.2), self.movementInfo['playery']

        self.basex = self.movementInfo['basex']
        self.baseShift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        # get 2 new pipes to add to upperPipes lowerPipes list
        self.newPipe1 = self.getRandomPipe()
        self.newPipe2 = self.getRandomPipe()

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
        self.playerVelY = -9   # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10   # max vel along Y, max descend speed
        self.playerMinVelY = -8   # min vel along Y, max ascend speed
        self.playerAccY = 1   # players downward accleration
        self.playerFlapAcc = -9   # players speed on flapping
        self.playerFlapped = False  # True when player flaps

        self.draw_frame_update()
        frame_set = self.get_frame_set()
        return frame_set

    def step(self, action):
        self.one_iter(action)

        frame_set = self.get_frame_set()

        if self.done:
            reward = 0
            print 'Game over please initialize'
        else:
            reward = 1

        return frame_set, reward, self.done

    def draw_frame_update(self):
        # draw sprites
        self.SCREEN.blit(self.IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            self.SCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            self.SCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        self.SCREEN.blit(self.IMAGES['base'], (self.basex, self.BASEY))
        # print score so player overlaps the score
        self.SCREEN.blit(self.IMAGES['player'][self.playerIndex], (self.playerx, self.playery))

        pygame.display.update()
        self.FPSCLOCK.tick(self.FPS)

    def one_iter(self, action):
        if self.done:
            return

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
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playerHeight = self.IMAGES['player'][self.playerIndex].get_height()
        self.playery += min(self.playerVelY, self.BASEY - self.playery - self.playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = self.getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -self.IMAGES['pipe'][0].get_width():
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        self.draw_frame_update()

        return 1, False

    def playerShm(self, playerShm):
        """oscillates the value of playerShm['val'] between 8 and -8"""
        if abs(playerShm['val']) == 8:
            playerShm['dir'] *= -1

        if playerShm['dir'] == 1:
            playerShm['val'] += 1
        else:
            playerShm['val'] -= 1

    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapY = random.randrange(0, int(self.BASEY * 0.6 - self.PIPEGAPSIZE))
        gapY += int(self.BASEY * 0.2)
        pipeHeight = self.IMAGES['pipe'][0].get_height()
        pipeX = self.SCREENWIDTH + 10

        return [
            {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
            {'x': pipeX, 'y': gapY + self.PIPEGAPSIZE},  # lower pipe
        ]

    def checkCrash(self, player, upperPipes, lowerPipes):
        """returns True if player collders with base or pipes."""
        pi = player['index']
        player['w'] = self.IMAGES['player'][0].get_width()
        player['h'] = self.IMAGES['player'][0].get_height()

        # if player crashes into ground
        if player['y'] + player['h'] >= self.BASEY - 1:
            return [True, True]
        else:

            playerRect = pygame.Rect(player['x'], player['y'],
                                     player['w'], player['h'])
            pipeW = self.IMAGES['pipe'][0].get_width()
            pipeH = self.IMAGES['pipe'][0].get_height()

            for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

                # player and upper/lower pipe hitmasks
                pHitMask = self.HITMASKS['player'][pi]
                uHitmask = self.HITMASKS['pipe'][0]
                lHitmask = self.HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = self.pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = self.pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return [True, False]

        return [False, False]

    def pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in xrange(rect.width):
            for y in xrange(rect.height):
                if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                    return True
        return False

    def getHitmask(self, image):
        """returns a hitmask using an image's alpha."""
        mask = []
        for x in range(image.get_width()):
            mask.append([])
            for y in range(image.get_height()):
                mask[x].append(bool(image.get_at((x, y))[3]))
        return mask


if __name__ == '__main__':
    env = FlappyBirdEnv()
    env.init_game()
    env.step(1)
    env.step(1)
    frame_set, _, _ = env.step(1)
    import pdb
    from scipy.misc import toimage
    for i in xrange(0, 4):
        toimage(frame_set[i]).show()
    pdb.set_trace()
    # while 1:
    #     done = False
    #     while not done:
    #         _, done = env.step(0)
    #     env.init_game()
