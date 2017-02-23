import numpy as np
__author__ = "zhiwei"


def epsilon_greedy_action_select(
    DQN_Q_approximator,
    state
):
    """
        Get the greedy action of a given state
        state: 4 frames
        action: [0, 1] or [1, 0] (np array)
    """
    Q_func = DQN_Q_approximator.eval(state)
    action = np.zeros(Q_func.shape())
    action[Q_func.index(max(Q_func))] = 1
    return action
