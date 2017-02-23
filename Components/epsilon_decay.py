def epsilon_decay(epsilon, epsilon_final, decay_rate=0.999):
    """
    decay the epsilon factor in greedy action selection with
    decay_rate until epsilon_final reached
    """
    if not 0 < decay_rate < 1:
        decay_rate = 0.999
    if epsilon <= epsilon_final:
        return epsilon_final
    return epsilon * decay_rate
