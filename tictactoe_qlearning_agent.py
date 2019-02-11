from rl_base.qlearning_agent import QLearningAgent
from tictactoe_action_helper import TicTacToeActionHelper

class TicTacToeQLearningAgent(QLearningAgent):
    def __init__(self):
        super(TicTacToeQLearningAgent, self).__init__()

    def _state_to_string(self, state):
        return TicTacToeActionHelper.state_to_string(state)

    def _get_available_actions(self, state):
        return TicTacToeActionHelper.get_available_actions(state)
