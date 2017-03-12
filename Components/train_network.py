import numpy as np
import pdb


def batch_parser(batch):
    states = []
    actions = []
    rewards = []
    states_bar = []
    dones = []
    for batch_instance in batch:
        states.append(batch_instance[0])
        actions.append(batch_instance[1])
        rewards.append(batch_instance[2])
        states_bar.append(batch_instance[3])
        dones.append(batch_instance[4])
    return np.array(states), np.array(actions), np.array(rewards), np.array(states_bar), dones


def train_network(
        DQN,
        batch,
        discount_factor=0.99,
        learning_rate=0.0001):
    # pdb.set_trace()t
    states, actions, rewards, states_bar, dones = batch_parser(batch)
    # pdb.set_trace()
    states_bar_predict_val = DQN.predict(states_bar)
    # pdb.set_trace()
    target_q_func = rewards + discount_factor * np.amax(states_bar_predict_val, axis=1)
    target_q_func[np.where(dones == 0)] = 0
    # pdb.set_trace()
    # cost = DQN.test_api(states, actions, target_q_func)
    # print cost
    cost = DQN.train_network(states, actions, target_q_func)
    return cost


def main():
    a = []
    for x in xrange(1, 10):
        a.append(np.array([0, 1]))
    a = np.array(a)
    print a.shape


if __name__ == '__main__':
    main()
