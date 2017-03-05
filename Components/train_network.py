import numpy as np


def batch_parser(batch):
    states = []
    actions = []
    rewards = []
    states_bar = []
    for batch_instance in batch:
        states.append(batch[0])
        actions.append(batch[1])
        rewards.append(batch[2])
        states_bar.append(batch[3])
    return np.array(states), np.array(actions), np.array(rewards), np.array(states_bar)


def reconstruct_label(actions, state_predict_val, td_error):
    pass


def train_network(
        DQN,
        batch,
        batch_parser,
        discount_factor=0.99,
        epsilon=0.0001,
        learning_rate=0.0001):
    states, actions, rewards, states_bar = batch_parser(batch)
    state_predict_val = DQN.predict(states)
    state_bar_predict_val = DQN.predict(states_bar)
    td_error = reward + discount_factor * np.max(state_bar_predict_val) - \
        actions * state_predict_val
    label = reconstruct_label(actions,
                              state_predict_val,
                              td_error)
    DQN.train_network(states, label)
