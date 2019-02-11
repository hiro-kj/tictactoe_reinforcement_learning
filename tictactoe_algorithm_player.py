import numpy as np
import random
from tictactoe_action_helper import TicTacToeActionHelper

class TicTacToeAlgorithmPlayer:
    def __init__(self, player_number):
        self.player_number = player_number

    def get_player_number(self):
        return self.player_number

    def select_action(self, state):
        raise NotImplementedError('Inheriting classes must override action.')

class TicTacToeDumbAlgorithmPlayer(TicTacToeAlgorithmPlayer):
    def __init__(self, player_number):
        super(TicTacToeDumbAlgorithmPlayer, self).__init__(player_number)

    def select_action(self, state):
        return TicTacToeActionHelper.pick_random_action(state)

class TicTacToeDecentAlgorithmPlayer(TicTacToeAlgorithmPlayer):
    def __init__(self, player_number, opponent_player_number):
        super(TicTacToeDecentAlgorithmPlayer, self).__init__(player_number)
        self.opponent_player_number = opponent_player_number

    def select_action(self, state):
        winning_action = TicTacToeActionHelper.select_winning_action(self.player_number, state)
        if (not (winning_action is None)):
            return winning_action
        
        defending_action = TicTacToeActionHelper.select_winning_action(self.opponent_player_number, state)
        if (not (defending_action is None)):
            return defending_action
        
        return TicTacToeActionHelper.pick_random_action(state)
