# Import modules
from FlappyBirdToolbox import FlappyBirdEnv
# Import function
from Components import Memory, DQN_Flappy_Bird
from Components import epsilon_greedy_action_select, train_network, epsilon_decay

import numpy as np

# Game set up
ACTION_NUM = 2

# Training set up
NUM_EPISODE = 1000000000
BATCH_SIZE = 32
MEMERY_LIMIT = 50000
FRAME_SETS_SIZE = 4

# RL parameters
BELLMAN_FACTOR = 0.99

# exploration setting
OBSERVE_PHASE = 50000
EXPLORE_PHASE = 100000

# epsilon setting
EPSILON_DISCOUNT = 0.9999
EPSILON_START = 0.5
EPSILON_FINAL = 0.0001


def train_dqn():
    epsilon = EPSILON_START

    env = FlappyBirdEnv('FlappyBirdToolbox/',
                        frame_set_size=FRAME_SETS_SIZE,
                        mode='train')

    # Create memory pool
    D = Memory(MEMERY_LIMIT, FRAME_SETS_SIZE)

    # Create network object
    DQN_Q_approximator = DQN_Flappy_Bird()

    DQN_Q_approximator.load_weights()

    # Q-Learning framework
    cost = 0
    total_step = 0
    for num_episode in xrange(NUM_EPISODE):
        state = env.init_game()

        done = False

        while not done:
            total_step += 1

            action = epsilon_greedy_action_select(
                DQN_Q_approximator,
                state,
                ACTION_NUM,
                epsilon
            )

            state_bar, reward, done = env.step(np.argmax(action))

            D.add((state, action, reward, state_bar, done))

            if total_step > OBSERVE_PHASE:
                batch = D.sample(BATCH_SIZE)

                cost = train_network(
                    DQN_Q_approximator,
                    batch,
                    BELLMAN_FACTOR
                )
            print "reward: ", reward, " cost: ", cost, " action: ", np.argmax(action), " if game continue: ", done, " epsilon: ", epsilon

            state = state_bar

        if 0 == (num_episode % 10000):
            print "Cost is: ", cost
            DQN_Q_approximator.save_weights(num_episode)  # save weights

        if total_step > EXPLORE_PHASE:
            epsilon = EPSILON_FINAL
        elif total_step > OBSERVE_PHASE:
            epsilon = epsilon_decay(epsilon, EPSILON_DISCOUNT, EPSILON_FINAL)


def main():
    train_dqn()


if __name__ == '__main__':
    train_dqn()
