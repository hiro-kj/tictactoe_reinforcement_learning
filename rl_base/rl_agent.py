import numpy as np
import random

class ReinforcementLearningAgent:
    def __init__(self, epsilon=1, alpha=0.5, gamma=1):
        self.alpha = alpha
        self.gamma = gamma
        self.q = {}
        self.epsilon = epsilon
    
    def _state_to_string(self, state):
        raise NotImplementedError('Inheriting classes must override _state_to_string.')

    def _get_available_actions(self, state):
        raise NotImplementedError('Inheriting classes must override _get_available_actions.')

    def get_epsilon(self):
        return self.epsilon

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
    
    def select_action(self, state):
        # choice = 0: get a max Q, 1: get a random action
        choice = np.random.binomial(1, self.epsilon)
        
        if choice == 0:
            _, action = self._get_best_value_and_action(state)
        else:
            actions = self._get_available_actions(state)
            action = random.choice(actions) if actions else None

        return action
    
    def _get_best_value(self, state):
        # pylint: disable=E1111
        state_str = self._state_to_string(state)
        # pylint: enable=E1111
        if state_str not in self.q:
            self.q[state_str] = {key: 0 for key in self._get_available_actions(state)}

        q = self.q[state_str]
        return state_str, q[max(q, key=lambda key: q[key])] if q else 0

    def _get_best_value_and_action(self, state):
        state_str, max_value = self._get_best_value(state)
        q = self.q[state_str]
        max_actions = [key for key in q if q[key] == max_value]
        action = random.choice(max_actions) if max_actions else None
        return max_value, action

    def learn(self):
        raise NotImplementedError('Inheriting classes must override learn.')

        