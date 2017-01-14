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
from FlappyBirdToolbox.FlappyBirdEnv import FlappyBirdEnv
from Memery import Memery
from DQN_Flappy_Bird import DQN_Flappy_Bird
from RLToolbox.epsilon_greedy_action_select import epsilon_greedy_action_select


def train_network(deep_Q_network, batch):
    pass


NUM_EPISODE = 1000000
Batch_SIZE = 128

env = FlappyBirdEnv()

D = Memery()

DQN_Q_approximator = DQN_Flappy_Bird()

DQN_Q_approximator.checkout()

for num_episode in xrange(NUM_EPISODE):
    frame_sets = env.init_game()
    done = False

    while not done:
        action = epsilon_greedy_action_select(
            DQN_Q_approximator,
            frame_sets
        )
        frame_sets_next, reward, done = env.step(action)

        D.add((frame_sets, action, reward, frame_sets_next))

        batch = D.sample(Batch_SIZE)

        train_network(DQN_Q_approximator, batch)
