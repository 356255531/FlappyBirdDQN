from FlappyBirdToolbox import FlappyBirdEnv
from Components import DQN_Flappy_Bird, greedy_action_select


def main():
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


if __name__ == '__main__':
    main()
