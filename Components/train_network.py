
def reconstruct_label(actions, state_predict_val, td_error):
    pass


def train_network(
        DQN,
        batch,
        batch_parser,
        discount_factor=0.99,
        epsilon=0.0001,
        learning_rate=0.0001):
    states, actions, reward, states_bar = batch_parser(batch)
    state_predict_val = DQN.predict(states)
    state_bar_predict_val = DQN.predict(states_bar)
    td_error = reward + discount_factor * max(state_bar_predict_val) - \
        actions * state_predict_val
    label = reconstruct_label(actions,
                              state_predict_val,
                              td_error)
    DQN.train_network(states, label)
