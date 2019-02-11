from rl_base.rl_experiment import QLearningExperiment
from rl_base.rl_experiment import SarsaExperiment
from tictactoe_env import TicTacToeEnv
from tictactoe_qlearning_agent import TicTacToeQLearningAgent
from tictactoe_sarsa_agent import TicTacToeSarsaAgent
from tictactoe_algorithm_player import TicTacToeDecentAlgorithmPlayer
from tictactoe_move_analyzer import TicTacToeMoveAnalyzer
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

    num_games_to_analyze = 100
    analyzer = TicTacToeMoveAnalyzer(moves, rewards, num_games_to_analyze)
    num_wins = analyzer.num_wins()
    num_corner_openings = analyzer.num_corner_openings()
    text = "{0} wins and {1} corner openings in the last {2} games.".format(num_wins, num_corner_openings, num_games_to_analyze)

    TicTacToePlotter.plot_episode_reward(rewards, text)

main()
