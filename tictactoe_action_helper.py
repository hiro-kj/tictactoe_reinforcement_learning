import numpy as np
import random

class TicTacToeActionHelper:
    @staticmethod
    def state_to_string(state):
        state_str = ""
        for i in range(9):
            state_str += str(state[i // 3, i % 3])
        return state_str

    @staticmethod
    def get_available_actions(state):
        available_actions = []
        for i in range(9):
            if state[i // 3, i % 3] == 0:
                available_actions.append(i)
        return available_actions
        
    @staticmethod
    def pick_random_action(state):
        available_actions = TicTacToeActionHelper.get_available_actions(state)
        if not available_actions:
            return None
        return random.choice(available_actions)

    @staticmethod
    def select_winning_action(player_number, state):
        winning_lines = [ [0, player_number, player_number], [player_number, 0, player_number], [player_number, player_number, 0] ]
        
        # test rows
        for i in range(3):
            row = state[i, :]
            for j in range(3):
                if (np.array_equal(row, winning_lines[j])):
                    return i * 3 + j

        # test columns
        for i in range(3):
            column = state[:, i]
            for j in range(3):
                if (np.array_equal(column, winning_lines[j])):
                    return j * 3 + i

        # test diagonals
        diagonal1 = [state[0, 0], state[1, 1], state[2, 2]]
        diagonal2 = [state[0, 2], state[1, 1], state[2, 0]]
        for i in range(3):
            if (np.array_equal(diagonal1, winning_lines[i])):
                return i * 3 + i
            if (np.array_equal(diagonal2, winning_lines[i])):
                return i * 3 + 2 - i
        
        # there are no winning lines
        return None
