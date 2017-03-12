# Import modules
from FlappyBirdToolbox import FlappyBirdEnv
# Import function
from Components import Memory, DQN_Flappy_Bird
from Components import image_preprocess, epsilon_greedy_action_select,\
    train_network, epsilon_decay

import pdb
import numpy as np
# Game set up
ACTION_NUM = 2

# Training set up
NUM_EPISODE = 10000000
BATCH_SIZE = 32
MEMERY_LIMIT = 50000
FRAME_SETS_SIZE = 4

# RL parameters
bellman_factor = 0.99

EPSILON_DISCOUNT = 0.99
EPSILON_START = 0.1
EPSILON_FINAL = 0.0001

epsilon = EPSILON_START


env = FlappyBirdEnv('FlappyBirdToolbox/',
                    frame_set_size=FRAME_SETS_SIZE,
                    mode='train')

D = Memory(MEMERY_LIMIT, FRAME_SETS_SIZE)

DQN_Q_approximator = DQN_Flappy_Bird(ACTION_NUM)

DQN_Q_approximator.load_weights()  # check if pre-trained weights exists

for num_episode in xrange(NUM_EPISODE):  # Q-Learning framework
    state = env.init_game()
    # pdb.set_trace()
    state = image_preprocess(state)
    # pdb.set_trace()
    done = False

    while not done:
        # pdb.set_trace()
        action = epsilon_greedy_action_select(
            DQN_Q_approximator,
            state,
            ACTION_NUM,
            epsilon
        )
        # pdb.set_trace()
        state_bar, reward, done = env.step(np.argmax(action))
        # pdb.set_trace()
        state_bar = image_preprocess(state_bar)
        # pdb.set_trace()
        D.add((state, action, reward, state_bar))

        batch = D.sample(BATCH_SIZE)
        # pdb.set_trace()
        cost = train_network(
            DQN_Q_approximator,
            batch,
            bellman_factor,
            epsilon
        )

        state = state_bar

    if 0 == (num_episode % 1000):
        print "Cost is: ", cost
        DQN_Q_approximator.save_weights(num_episode)  # save weights
        epsilon = epsilon_decay(epsilon, EPSILON_DISCOUNT, EPSILON_FINAL)
