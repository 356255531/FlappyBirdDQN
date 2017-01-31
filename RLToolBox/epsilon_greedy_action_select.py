__author__ = "zhiwei"


def epsilon_greedy_action_select(
    DQN_Q_approximator,
    frame_sets
):
    Q_func = DQN_Q_approximator.predict(frame_sets)

    if Q_func[0] >= Q_func[1]:  # Hard code
        return False
    else:
        return True
