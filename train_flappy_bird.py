# Import modules
from FlappyBirdToolbox import FlappyBirdEnv
# Import function
from Components import Memory, DQN_Flappy_Bird
from Components import image_preprocess, epsilon_greedy_action_select,\
    train_network, epsilon_decay

# Game set up
ACTION_NUM = 2

# Training set up
NUM_EPISODE = 1000000
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
    state = image_preprocess(state)

    done = False

    while not done:
        action = epsilon_greedy_action_select(DQN_Q_approximator,
                                              state,
                                              ACTION_NUM,
                                              epsilon)

        state_bar, reward, done = env.step(action.index(1))
        state_bar = image_preprocess(state_bar)

        D.add((state, action, reward, state_bar))

        batch = D.sample(BATCH_SIZE)

        train_network(DQN_Q_approximator,
                      batch,
                      bellman_factor,
                      epsilon)

    if 0 == (num_episode % 1):
        DQN_Q_approximator.save_weights(num_episode)  # save weights
        epsilon = epsilon_decay(epsilon, EPSILON_DISCOUNT, EPSILON_FINAL)
