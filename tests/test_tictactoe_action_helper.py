import numpy as np
import pytest
from unittest.mock import MagicMock
from tictactoe_action_helper import TicTacToeActionHelper

def test_state_to_string():
    state = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
    state_str = TicTacToeActionHelper.state_to_string(state)
    assert state_str == "100020001"

    state = np.array([[1, 1, 2], [2, 2, 1], [1, 2, 1]])
    state_str = TicTacToeActionHelper.state_to_string(state)
    assert state_str == "112221121"

    state = np.zeros((3, 3), dtype=np.int)
    state_str = TicTacToeActionHelper.state_to_string(state)
    assert state_str == "000000000"


def test_get_available_action():
    state = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
    available_actions = TicTacToeActionHelper.get_available_actions(state)
    assert available_actions == [1, 2, 3, 5, 6, 7]

    state = np.array([[0, 2, 1], [2, 0, 1], [2, 1, 0]])
    available_actions = TicTacToeActionHelper.get_available_actions(state)
    assert available_actions == [0, 4, 8]

    # none available
    state = np.array([[1, 1, 2], [2, 2, 1], [1, 2, 1]])
    available_actions = TicTacToeActionHelper.get_available_actions(state)
    assert available_actions == []

    # all available
    state = np.zeros((3, 3), dtype=np.int)
    available_actions = TicTacToeActionHelper.get_available_actions(state)
    assert available_actions == [0, 1, 2, 3, 4, 5, 6, 7, 8]

def test_pick_random_action(monkeypatch):
    # make random choice always pick the middle element.
    mock_random_choice = MagicMock(side_effect= lambda x: x[(len(x) + 1) // 2 - 1] if len(x) > 0 else None)
    monkeypatch.setattr("random.choice", mock_random_choice)

    state = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0]])
    assert TicTacToeActionHelper.pick_random_action(state) == 5
    
    state = np.array([[1, 1, 2], [2, 2, 1], [2, 0, 1]])
    assert TicTacToeActionHelper.pick_random_action(state) == 7

    # no actions to pick
    state = np.array([[1, 1, 2], [2, 2, 1], [2, 1, 1]])
    assert TicTacToeActionHelper.pick_random_action(state) is None

def test_select_winning_action():
    # test rows
    state = np.array([[1, 1, 0],
                      [2, 0, 2],
                      [1, 2, 0]])
    assert TicTacToeActionHelper.select_winning_action(1, state) == 2
    assert TicTacToeActionHelper.select_winning_action(2, state) == 4

    # test columns
    state = np.array([[0, 0, 1],
                      [2, 2, 1],
                      [2, 1, 0]])
    assert TicTacToeActionHelper.select_winning_action(1, state) == 8
    assert TicTacToeActionHelper.select_winning_action(2, state) == 0

    # test diagnals
    state = np.array([[1, 2, 2],
                      [0, 1, 0],
                      [0, 0, 0]])
    assert TicTacToeActionHelper.select_winning_action(1, state) == 8

    state = np.array([[2, 0, 0],
                      [0, 1, 0],
                      [1, 0, 2]])
    assert TicTacToeActionHelper.select_winning_action(1, state) == 2

    # test no winning lines
    state = np.array([[1, 2, 1],
                      [1, 2, 2],
                      [2, 1, 1]])
    assert TicTacToeActionHelper.select_winning_action(1, state) is None
    assert TicTacToeActionHelper.select_winning_action(2, state) is None

    state = np.array([[1, 0, 0],
                      [0, 2, 0],
                      [0, 0, 0]])
    assert TicTacToeActionHelper.select_winning_action(1, state) is None
    assert TicTacToeActionHelper.select_winning_action(2, state) is None
