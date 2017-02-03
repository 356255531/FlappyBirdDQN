# Initialize replay memory D to size N
# Initialize action-value function Q with random weights
# for episode = 1, M do
#     Initialize state s_1
#     for t = 1, T do
#         With probability ϵ select random action a_t
#         otherwise select a_t=max_a  Q(s_t,a; θ_i)
#         Execute action a_t in emulator and observe r_t and s_(t+1)
#         Store transition (s_t,a_t,r_t,s_(t+1)) in D
#         Sample a minibatch of transitions (s_j,a_j,r_j,s_(j+1)) from D
#         Set y_j:=
#             r_j for terminal s_(j+1)
#             r_j+γ*max_(a^' )  Q(s_(j+1),a'; θ_i) for non-terminal s_(j+1)
#         Perform a gradient step on (y_j-Q(s_j,a_j; θ_i))^2 with respect to θ
#     end for
# end for
# Import modules
from FlappyBirdToolbox import FlappyBirdEnv
from Components import Memery, DQN_Flappy_Bird

# Import function
from RLToolbox.Components import epsilon_greedy_action_select, train_network, image_pre_process

# Training set up
NUM_EPISODE = 1000000
BATCH_SIZE = 32
MEMERY_LIMIT = 50000
FRAME_SETS_SIZE = 4

# RL parameters
discount_facotr = 0.99
epsilon_start = 0.1
epsilon_final = 0.0001


def main():
    env = FlappyBirdEnv()

    D = Memery(MEMERY_LIMIT, FRAME_SETS_SIZE)

    DQN_Q_approximator = DQN_Flappy_Bird()

    DQN_Q_approximator.check_weights()  # check if pre-trained weights exists

    for num_episode in xrange(NUM_EPISODE):  # Q-Learning framework
        frame_sets = env.init_game()
        frame_sets = image_pre_process(frame_sets)

        done = False

        epsilon = epsilon_final - (epsilon_start - epsilon_final) / (num_episode * 100.0)
        while not done:
            action = epsilon_greedy_action_select(
                DQN_Q_approximator,
                frame_sets
            )
            frame_sets_next, reward, done = env.step(action)
            frame_sets_next = image_pre_process(frame_sets_next)

            D.add((frame_sets, action, reward, frame_sets_next))

            batch = D.sample(BATCH_SIZE)

            DQN_Q_approximator.train_network(
                batch, discount_facotr, epsilon
            )

        if 0 == num_episode % 10000:
            DQN_Q_approximator.save_weights()  # save weights


if __name__ == '__main__':
    main()
