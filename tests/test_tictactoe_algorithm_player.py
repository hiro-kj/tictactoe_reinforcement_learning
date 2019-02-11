import pytest
import numpy as np
from tictactoe_algorithm_player import TicTacToeDumbAlgorithmPlayer
from tictactoe_algorithm_player import TicTacToeDecentAlgorithmPlayer

def test_dumb_algorithm_player_select_action():
    algorithm_player = TicTacToeDumbAlgorithmPlayer(2)

    state = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
    action = algorithm_player.select_action(state)
    assert action in [1, 2, 3, 5, 6, 7]

    state = np.array([[0, 2, 1], [2, 0, 1], [2, 1, 0]])
    action = algorithm_player.select_action(state)
    assert action in [0, 4, 8]

    # none available
    state = np.array([[1, 1, 2], [2, 2, 1], [1, 2, 1]])
    action = algorithm_player.select_action(state)
    assert action is None
    
    # all available
    state = np.zeros((3, 3), dtype=np.int)
    action = algorithm_player.select_action(state)
    assert action in [0, 1, 2, 3, 4, 5, 6, 7, 8]

def test_decent_algorithm_player_select_action():
    algorithm_player = TicTacToeDecentAlgorithmPlayer(2, 1)

    # pick randomly
    state = np.array([[1, 0, 0],
                      [0, 2, 0],
                      [0, 0, 1]])
    action = algorithm_player.select_action(state)
    assert action in [1, 2, 3, 5, 6, 7]

    # win it
    state = np.array([[1, 1, 2],
                      [0, 2, 0],
                      [0, 0, 1]])
    action = algorithm_player.select_action(state)
    assert action == 6

    # win it instead of defending
    state = np.array([[1, 0, 2],
                      [1, 0, 2],
                      [0, 1, 0]])
    action = algorithm_player.select_action(state)
    assert action == 8

    # defend
    state = np.array([[1, 0, 0],
                      [0, 2, 0],
                      [1, 0, 0]])
    action = algorithm_player.select_action(state)
    assert action == 3
    
    state = np.array([[1, 1, 0],
                      [0, 2, 0],
                      [0, 0, 0]])
    action = algorithm_player.select_action(state)
    assert action == 2

    # no actions to pick
    state = np.array([[1, 1, 2],
                      [2, 2, 1],
                      [1, 2, 1]])
    action = algorithm_player.select_action(state)
    assert action is None
