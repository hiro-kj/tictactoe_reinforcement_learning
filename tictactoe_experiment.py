import pandas as pd
from matplotlib import pyplot as plt

class TicTacToeExperiment:
    def __init__(self, env, agent, runner):
        self.env = env
        self.agent = agent
        self.runner = runner
        self.rewards = []
    
    def run_episodes(self, num_episodes, before_episode_callback = None, after_episode_callback = None):

        for episode_number in range(num_episodes):
            if not (before_episode_callback is None):
                before_episode_callback(self.env, self.agent, episode_number)

            total_reward = self.runner.run()
            self.rewards.append(total_reward)

            if not (after_episode_callback is None):
                after_episode_callback(self.env, self.agent, episode_number)    