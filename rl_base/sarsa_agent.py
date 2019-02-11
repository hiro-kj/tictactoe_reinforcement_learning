from rl_base.rl_agent import ReinforcementLearningAgent

class SarsaAgent(ReinforcementLearningAgent):
    def __init__(self, epsilon=1, alpha=0.5, gamma=1):
        super(SarsaAgent, self).__init__(epsilon, alpha, gamma)
    

    def learn(self, state1, action1, reward, state2, action2):
        # pylint: disable=E1111
        state_str1 = self._state_to_string(state1)
        state_str2 = self._state_to_string(state2)
        # pylint: enable=E1111

        if state_str1 not in self.q:
            self.q[state_str1] = {key: 0 for key in self._get_available_actions(state1)}

        if state_str2 not in self.q:
            self.q[state_str2] = {key: 0 for key in self._get_available_actions(state2)}

        q = self.q[state_str1][action1]
        next_q = self.q[state_str2][action2] if not (action2 is None) else 0 

        self.q[state_str1][action1] = q + self.alpha * (reward + self.gamma * next_q - q) 

        