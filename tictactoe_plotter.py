import pandas as pd
from matplotlib import pyplot as plt

class TicTacToePlotter:

    @staticmethod
    def plot_episode_reward(rewards, text, smoothing_window=50):

        wins_in_last_100_games = sum([r for r in rewards[-100:] if r == 1])

        fig = plt.figure(figsize=(10,5))
        rewards_smoothed = pd.Series(rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed)
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward (Smoothed)")
        plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
        plt.text(len(rewards), 0.5, text, fontsize=12, horizontalalignment='right', verticalalignment='center')
        plt.show(fig)

        return fig