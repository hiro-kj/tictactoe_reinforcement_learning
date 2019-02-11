from rl_base.sarsa_agent import SarsaAgent
from tictactoe_action_helper import TicTacToeActionHelper

class TicTacToeSarsaAgent(SarsaAgent):
    def __init__(self):
        super(TicTacToeSarsaAgent, self).__init__()

    def _state_to_string(self, state):
        return TicTacToeActionHelper.state_to_string(state)

    def _get_available_actions(self, state):
        return TicTacToeActionHelper.get_available_actions(state)
