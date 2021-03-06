import pandas as pd
from matplotlib import pyplot as plt

class TicTacToePlotter:

    @staticmethod
    def plot_episode_reward(rewards, text, rl_method, smoothing_window=50):
        fig = plt.figure(figsize=(10, 5))
        rewards_smoothed = pd.Series(rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed)
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward (Smoothed)")
        plt.title("Episode Reward over Time (Smoothed over window size {0}) with {1}".format(smoothing_window, rl_method))
        plt.text(len(rewards), 0, text, fontsize=12, horizontalalignment='right', verticalalignment='center')
        plt.show(fig)

        return fig