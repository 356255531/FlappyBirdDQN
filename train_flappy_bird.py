# Import modules
from FlappyBirdToolbox import FlappyBirdEnv
# Import function
# from Components import epsilon_greedy_action_select, train_network, epsilon_decay, Memery


# Training set up
NUM_EPISODE = 1000000
BATCH_SIZE = 32
MEMERY_LIMIT = 50000
FRAME_SETS_SIZE = 4

# RL parameters
discount_facotr = 0.99
epsilon_start = 0.1
epsilon_final = 0.0001

epsilon = epsilon_start


def main():
    env = FlappyBirdEnv('FlappyBirdToolbox/', mode='train')
    while 1:
        env.init_game()
        done = False
        while not done:
            _, _, done = env.step(1)
    # D = Memery(MEMERY_LIMIT, FRAME_SETS_SIZE)

    # DQN_Q_approximator = DQN_Flappy_Bird()

    # DQN_Q_approximator.load_weights()  # check if pre-trained weights exists

    # for num_episode in xrange(NUM_EPISODE):  # Q-Learning framework
    #     frame_sets = env.init_game(preprocess=True)

    #     done = False

    #     while not done:
    #         action = epsilon_greedy_action_select(DQN_Q_approximator, frame_sets)

    #         frame_sets_next, reward, done = env.step(action, preprocess=True)

    #         D.add((frame_sets, action, reward, frame_sets_next))

    #         batch = D.sample(BATCH_SIZE)

    #         train_ne2twork(DQN_Q_approximator, batch, discount_facotr, epsilon)

    #     if 0 == num_episode % 10000:
    #         DQN_Q_approximator.save_weights()  # save weights
    #         epsilon_decay(epsilon, epsilon_final)


if __name__ == '__main__':
    main()
