from rl_base.rl_experiment import QLearningExperiment
from rl_base.rl_experiment import SarsaExperiment
from tictactoe_env import TicTacToeEnv
from tictactoe_qlearning_agent import TicTacToeQLearningAgent
from tictactoe_sarsa_agent import TicTacToeSarsaAgent
from tictactoe_algorithm_player import TicTacToeDecentAlgorithmPlayer
from tictactoe_plotter import TicTacToePlotter

class Epsilon:
    def __init__(self, value, decay_rate):
        self.value = value
        self.decay_rate = decay_rate

    def decay(self):
        self.value = self.value * self.decay_rate

def main():
    num_episodes = 3000
    env = TicTacToeEnv(TicTacToeDecentAlgorithmPlayer(2, 1))
    agent = TicTacToeQLearningAgent()
    # agent = TicTacToeSarsaAgent()
    epsilon = Epsilon(1.0, 0.95)
    rewards = []
    moves = []
    def before_episode_callback(env, agent, episode_number):
        agent.set_epsilon(epsilon.value)
    def after_episode_callback(env, agent, episode_number, reward):
        epsilon.decay()
        rewards.append(reward)
        moves.append(env.get_moves())

    experiment = QLearningExperiment(env, agent, before_episode_callback, after_episode_callback)
    # experiment = SarsaExperiment(env, agent, before_episode_callback, after_episode_callback)

    experiment.experiment(num_episodes)

    TicTacToePlotter.plot_episode_reward(rewards)

main()
