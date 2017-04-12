from itertools import cycle
import random
import numpy as np
import cv2

import pygame
from pygame.locals import *


class FlappyBirdEnv(object):
    """
    The flappy bird simulator for DRL algorithm
    API:
        init_game(preprocess=False):
            reset game to initial state (if perform learning)
            return: the initial state (n frames here n = 4)

        step(action):
            take the action in the simulator
            return:
                frame_set(state): np array,
                reward,
                done(if game over)
    """

    def __init__(self, rel_path='', frame_set_size=4, mode='play'):
        super(FlappyBirdEnv, self).__init__()

        # The number of continous frame stream in one state
        self.frame_set_size = frame_set_size

        # Set a higher frequency in training process
        if mode == 'play':
            self.FPS = 30
        else:
            self.FPS = 30000

        # Display information
        self.SCREENWIDTH = 288
        self.SCREENHEIGHT = 512

        # amount by which base can maximum shift to left
        self.PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
        self.BASEY = self.SCREENHEIGHT * 0.79
        # image and hitmask  dicts
        self.IMAGES, self.HITMASKS = {}, {}

        # list of all possible players (tuple of 3 positions of flap)
        self.PLAYERS_PATH = (
            rel_path + 'assets/sprites/redbird-upflap.png',
            rel_path + 'assets/sprites/redbird-midflap.png',
            rel_path + 'assets/sprites/redbird-downflap.png'
        )

        # list of backgrounds
        self.BACKGROUNDS_PATH = rel_path + 'assets/sprites/background-day.png'

        # list of pipes
        self.PIPES_PATH = rel_path + 'assets/sprites/pipe-green.png'

        # Initialize the display
        pygame.init()
        self.FPSCLOCK = pygame.time.Clock()
        self.SCREEN = pygame.display.set_mode((self.SCREENWIDTH, self.SCREENHEIGHT))
        pygame.display.set_caption('Flappy Bird')

        # Load the graphical components
        self.IMAGES['base'] = pygame.image.load(rel_path + 'assets/sprites/base.png').convert_alpha()

        self.IMAGES['background'] = pygame.image.load(self.BACKGROUNDS_PATH).convert()

        # select random player sprites
        self.IMAGES['player'] = (
            pygame.image.load(self.PLAYERS_PATH[0]).convert_alpha(),
            pygame.image.load(self.PLAYERS_PATH[1]).convert_alpha(),
            pygame.image.load(self.PLAYERS_PATH[2]).convert_alpha(),
        )

        # select random pipe sprites
        self.IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(self.PIPES_PATH).convert_alpha(), 180),
            pygame.image.load(self.PIPES_PATH).convert_alpha(),
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

        # get the size of play and pipes
        self.PLAYER_WIDTH = self.IMAGES['player'][0].get_width()
        self.PLAYER_HEIGHT = self.IMAGES['player'][0].get_height()
        self.PIPE_WIDTH = self.IMAGES['pipe'][0].get_width()
        self.PIPE_HEIGHT = self.IMAGES['pipe'][0].get_height()
        self.BACKGROUND_HEIGHT = self.IMAGES['background'].get_height()

    def get_display_colored_image(self):
        frame = pygame.surfarray.array3d(
            pygame.display.get_surface()).T
        frame = np.swapaxes(frame, 0, 2)
        frame = np.swapaxes(frame, 0, 1)
        return frame

    def image_preprocess(self, colored_image):
        colored_image = colored_image[0:399, :, :]
        greyscale_image = cv2.cvtColor(
            cv2.resize(
                colored_image, (80, 80)), cv2.COLOR_BGR2GRAY)
        # pdb.set_trace()
        _, greyscale_image = cv2.threshold(greyscale_image, 20, 255, cv2.THRESH_BINARY)
        return greyscale_image

    def init_game(self):
        """
        Init the game and render the start frames
        return: initial frame set (state): numpy array
        """
        self.done = False

        # index of player to blit on screen
        playerIndexGen = cycle([0, 1, 2, 1])
        # iterator used to change playerIndex after every 5th iteration

        playery = int((self.SCREENHEIGHT - self.PLAYER_HEIGHT) / 2)

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

        frame = self.get_display_colored_image()

        frame = self.image_preprocess(frame)

        self.frame_set = np.stack((frame, frame, frame, frame), axis=2)

        return self.frame_set

    def step(self, action):
        """
        Take action and feedback
        Return:
            Frame_set (state): numpy array in list
            reward:integer,
            if done: bool
        """
        if action not in [0, 1]:
            raise ValueError("action is either 0 or 1")

        if self.done:
            raise ValueError("Game over please initialize")

        reward = 0.1

        self.one_frame_move(action)
        self.one_frame_move(0)
        self.one_frame_move(0)
        self.one_frame_move(0)

        playerMidPos = self.playerx + self.PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + self.PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                # self.score += 1
                # SOUNDS['point'].play()
                reward = 1

        if self.done:
            reward = -1

        frame = self.get_display_colored_image()
        frame = self.image_preprocess(frame)
        frame = np.reshape(frame, (80, 80, 1))
        self.frame_set = np.append(frame, self.frame_set[:, :, :3], axis=2)

        return self.frame_set, reward, self.done

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

    def one_frame_move(self, action):
        if self.done:
            return
        crashTest = self.checkCrash(
            {
                'x': self.playerx,
                'y': self.playery,
                'index': self.playerIndex
            },
            self.upperPipes,
            self.lowerPipes
        )
        if crashTest[0] or crashTest[1]:
            self.done = True
            return

        if 1 == action:
            if self.SCREENHEIGHT - self.PLAYER_HEIGHT > self.playery and self.playery > -2 * self.PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

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
        self.playery += min(self.playerVelY, self.BASEY - self.playery - self.PLAYER_HEIGHT)

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
    env = FlappyBirdEnv('',
                        frame_set_size=4,
                        mode="play")
    while 1:
        done = False
        env.init_game()
        while not done:
            action = 0
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                    action = 1
            _, reward, done = env.step(action)
            print reward
