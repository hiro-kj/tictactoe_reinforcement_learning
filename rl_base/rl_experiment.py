
class ReinforcementLearningExperiment:
    def __init__(self, env, agent, before_episode_callback=None, after_episode_callback=None):
        self.env = env
        self.agent = agent
        self.before_episode_callback = before_episode_callback
        self.after_episode_callback = after_episode_callback

    def run_episode(self):
        raise NotImplementedError('Inheriting classes must override _state_to_string.')

    def experiment(self, num_episodes):
        for episode_number in range(num_episodes):
            if not (self.before_episode_callback is None):
                self.before_episode_callback(self.env, self.agent, episode_number)

            total_reward = self.run_episode()

            if not (self.after_episode_callback is None):
                self.after_episode_callback(self.env, self.agent, episode_number, total_reward)

class QLearningExperiment(ReinforcementLearningExperiment):
    def __init__(self, env, agent, before_episode_callback=None, after_episode_callback=None):
        super(QLearningExperiment, self).__init__(env, agent, before_episode_callback, after_episode_callback)

    def run_episode(self):
        state, is_done = self.env.reset()
        total_reward = 0
        
        while not is_done:
            action = self.agent.select_action(state)
            next_state, reward, is_done = self.env.step(action)
            
            self.agent.learn(state, action, reward, next_state)

            state = next_state
            total_reward += reward
        return total_reward

class SarsaExperiment(ReinforcementLearningExperiment):
    def __init__(self, env, agent, before_episode_callback=None, after_episode_callback=None):
        super(SarsaExperiment, self).__init__(env, agent, before_episode_callback, after_episode_callback)

    def run_episode(self):
        state, is_done = self.env.reset()
        total_reward = 0
        
        action = self.agent.select_action(state)
        
        while not is_done:
            next_state, reward, is_done = self.env.step(action)
            next_action = self.agent.select_action(next_state)

            self.agent.learn(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action
            total_reward += reward
        return total_reward
