
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """
    
    Q = 0
    for next_state in mdp.get_next_states(state, action):
        Q += mdp.get_transition_prob(state, action, next_state) * (mdp.get_reward(state, action, next_state)
                                                                   + gamma * state_values[next_state])
        
    return Q
