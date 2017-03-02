def epsilon_decay(epsilon, epsilon_final, decay_rate=0.999):
    """
    decay the epsilon factor in greedy action selection with
    decay_rate until epsilon_final reached
    """
    if epsilon == epsilon_final:
        return epsilon_final

    if epsilon < epsilon_final:
        return epsilon_final

    if epsilon > 1:
        return 0.1

    if not 0 < decay_rate < 1:
        decay_rate = 0.999

    return epsilon * decay_rate
