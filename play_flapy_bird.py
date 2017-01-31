from FlappyBirdToolbox.FlappyBirdEnv import FlappyBirdEnv
from DQN_Flappy_Bird import DQN_Flappy_Bird

# Import function
from RLToolbox.epsilon_greedy_action_select import greedy_action_select

env = FlappyBirdEnv()

DQN_Q_approximator = DQN_Flappy_Bird()

DQN_Q_approximator.check_weights()  # check if pre-trained weights exists

while 1:
    frame_sets = env.init_game()
    done = False

    while not done:
        action = greedy_action_select(
            DQN_Q_approximator,
            frame_sets
        )
        frame_sets_next, reward, done = env.step(action)

        env.render()
