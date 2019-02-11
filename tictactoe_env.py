import numpy as np
import random

PLAYER1 = 1
PLAYER2 = 2

class TicTacToeEnv:
    def __init__(self, opponent):
        self.board = None
        self.opponent = opponent
        self.moves = None

    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.int)
        is_done = False
        self.moves = []
        return self.get_state(), is_done  
        
    def step(self, action):
        if self._play(PLAYER1, action):
            return (self.get_state(), 1, True)

        if len(self.moves) == 9:
            return (self.get_state(), 0, True)

        opponents_action = self.opponent.select_action(self.get_state())
        if self._play(self.opponent.get_player_number(), opponents_action):
            return (self.get_state(), -1, True)

        return (self.get_state(), 0, False)

    def _did_win(self, player):
        line = np.full(3, player)
        for i in range(3):
            # test rows
            if np.array_equal(self.board[i, :], line):
                return True
            # test columns
            if np.array_equal(self.board[:, i], line):
                return True
        # test diagonals
        if self.board[1, 1] == player:
            if self.board[0, 0] == player and self.board[2, 2] == player:
                return True
            if self.board[0, 2] == player and self.board[2, 0] == player:
                return True
        return False
    
    def _play(self, player_number, action):
        if self.board[action // 3, action % 3] != 0:
            raise ValueError("The action is not available")

        self.board[action // 3, action % 3] = player_number
        self.moves.append(action)
        if self._did_win(player_number):
            return True
        return False

    def get_state(self):
        return self.board.copy()

    def get_moves(self):
        return self.moves.copy()
