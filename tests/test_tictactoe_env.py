import pytest
import numpy as np
from unittest.mock import Mock 
from tictactoe_env import TicTacToeEnv

PLAYER1 = 1
PLAYER2 = 2

def test_reset():
    mock_opponent = Mock()
    env = TicTacToeEnv(mock_opponent)
    state, is_done = env.reset()
    assert np.array_equal(state, np.zeros((3, 3), dtype=np.int))
    assert is_done == False

def test_did_win():
    mock_opponent = Mock()
    env = TicTacToeEnv(mock_opponent)

    for player_a, player_b in [(PLAYER1, PLAYER2), (PLAYER2, PLAYER1)]:
        # test rows
        for i in range(3):
            env.reset()
            env.board[i, :] = player_a
            assert env._did_win(player_a)
            assert not env._did_win(player_b)

        # test columns
        for i in range(3):
            env.reset()
            env.board[:, i] = player_a
            assert env._did_win(player_a)
            assert not env._did_win(player_b)

        # test diagonals
        env.reset()
        env.board[0, 0] = player_a
        env.board[1, 1] = player_a
        env.board[2, 2] = player_a
        assert env._did_win(player_a)
        assert not env._did_win(player_b)
    
        env.reset()
        env.board[0, 2] = player_a
        env.board[1, 1] = player_a
        env.board[2, 0] = player_a
        assert env._did_win(player_a)
        assert not env._did_win(player_b)

    # test no winners
    env.reset()
    assert not env._did_win(PLAYER1)
    assert not env._did_win(PLAYER2)

    env.board[0, 0] = PLAYER1
    assert not env._did_win(PLAYER1)
    assert not env._did_win(PLAYER2)

    env.board[0, 1] = PLAYER1
    assert not env._did_win(PLAYER1)
    assert not env._did_win(PLAYER2)

    env.board[0, 2] = PLAYER2
    assert not env._did_win(PLAYER1)
    assert not env._did_win(PLAYER2)

    env.board[1, 0] = PLAYER2
    assert not env._did_win(PLAYER1)
    assert not env._did_win(PLAYER2)

    env.board[1, 1] = PLAYER2
    assert not env._did_win(PLAYER1)
    assert not env._did_win(PLAYER2)

    env.board[1, 2] = PLAYER1
    assert not env._did_win(PLAYER1)
    assert not env._did_win(PLAYER2)

    env.board[2, 0] = PLAYER1
    assert not env._did_win(PLAYER1)
    assert not env._did_win(PLAYER2)

    env.board[2, 1] = PLAYER2
    assert not env._did_win(PLAYER1)
    assert not env._did_win(PLAYER2)

    env.board[2, 2] = PLAYER2
    assert not env._did_win(PLAYER1)
    assert not env._did_win(PLAYER2)

def _play_out(agent_actions, opponent_actions, expected_reward):
    mock_opponent = Mock()
    env = TicTacToeEnv(mock_opponent)
    _, is_done = env.reset()

    attrs = { "select_action.side_effect": opponent_actions, "get_player_number.return_value" : 2 }
    mock_opponent.configure_mock(**attrs)
    
    expected_state = np.zeros((3, 3), dtype=np.int)
    i = 0

    # call env.step until the end
    while not is_done:

        state, reward, is_done = env.step(agent_actions[i])

        expected_state[agent_actions[i] // 3, agent_actions[i] % 3] = PLAYER1
        if i < len(opponent_actions):
            expected_state[opponent_actions[i] // 3, opponent_actions[i] % 3] = PLAYER2
        if not is_done:
            assert np.array_equal(state, expected_state)
            assert reward == 0
            mock_opponent.select_action.assert_called_once()
            mock_opponent.reset_mock()
        i += 1

    assert np.array_equal(state, expected_state)
    assert i == len(agent_actions)
    assert reward == expected_reward
    if i > len(opponent_actions):
        mock_opponent.select_action.assert_not_called()
    else:
        mock_opponent.select_action.assert_called_once()

def test_step_agent_wins():
    _play_out([0, 1, 2], [3, 4], 1)

def test_step_agent_loses():
    _play_out([0, 1, 6], [3, 4, 5], -1)

def test_step_agent_ties():
    _play_out([0, 1, 5, 6, 8], [3, 4, 2, 7], 0)

def test_get_moves():
    mock_opponent = Mock()
    env = TicTacToeEnv(mock_opponent)
    env.reset()
    assert not env.get_moves()

    attrs = { "select_action.side_effect": [3, 4, 2, 7], "get_player_number.return_value" : 2 }
    mock_opponent.configure_mock(**attrs)
    
    env.step(0)
    assert env.get_moves() == [0, 3]

    env.step(1)
    env.step(5)
    env.step(6)
    env.step(8)
    assert env.get_moves() == [0, 3, 1, 4, 5, 2, 6, 7, 8]

    env.reset()
    assert not env.get_moves()

def test_step_with_action_taken_before():
    mock_opponent = Mock()
    env = TicTacToeEnv(mock_opponent)
    env.reset()

    attrs = { "select_action.side_effect": [1, 3], "get_player_number.return_value" : 2 }
    mock_opponent.configure_mock(**attrs)
    
    env.step(0)
    with pytest.raises(Exception):
        env.step(0)

def test_step_with_action_taken_by_opponent():
    mock_opponent = Mock()
    env = TicTacToeEnv(mock_opponent)
    env.reset()

    attrs = { "select_action.side_effect": [1, 3], "get_player_number.return_value" : 2 }
    mock_opponent.configure_mock(**attrs)
    
    env.step(0)
    with pytest.raises(Exception):
        env.step(1)
    