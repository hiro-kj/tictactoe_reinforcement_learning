import pytest
from tictactoe_move_analyzer import TicTacToeMoveAnalyzer

@pytest.fixture
def moves():
    return [                            # 11 moves
        [0, 1, 4, 8, 6, 2, 3],          # AI Moves: Corner->Center->Win
        [0, 4, 8, 1, 7, 2, 6, 3, 5],    # AI Moves: Corner->Cornder->Tie
        [4, 8, 0, 1, 3, 5, 6],          # AI Moves: Center->Corner->Win
        [0, 4, 8, 2, 6, 7, 3],          # AI Moves: Corner->Corner->Win
        [2, 7, 4, 6, 8, 5, 0],          # AI Moves: Corner->Center->Win
        [4, 0, 8, 6, 2, 3],             # AI Moves: Center->Corner->Lose
        [0, 1, 4, 8, 6, 2, 3],          # AI Moves: Corner->Center->Win (same moves as 0)
        [0, 4, 8, 1, 7, 2, 6, 3, 5],    # AI Moves: Corner->Cornder->Tie (same moves as 1)
        [4, 8, 0, 1, 3, 5, 6],          # AI Moves: Center->Corner->Win (same moves as 2)
        [0, 4, 8, 2, 6, 7, 3],          # AI Moves: Corner->Corner->Win (same moves as 3)
        [2, 7, 4, 6, 8, 5, 0]           # AI Moves: Corner->Center->Win (same moves as 4)
    ]

@pytest.fixture
def rewards():
    return [1, 0, 1, 1, 1, -1, 1, 0, 1, 1, 1]

@pytest.fixture
def moves_won():
    return [    # 8 wins
        [0, 1, 4, 8, 6, 2, 3],
        [4, 8, 0, 1, 3, 5, 6],
        [0, 4, 8, 2, 6, 7, 3],
        [2, 7, 4, 6, 8, 5, 0],
        [0, 1, 4, 8, 6, 2, 3],
        [4, 8, 0, 1, 3, 5, 6],
        [0, 4, 8, 2, 6, 7, 3],
        [2, 7, 4, 6, 8, 5, 0]
    ]


def test_moves_won(moves, rewards, moves_won):
    # wins [0, 2, 3, 4, 6, 8, 9, 10]
    analyzer = TicTacToeMoveAnalyzer(moves, rewards, 10)
    assert analyzer.moves_won == moves_won[1:]
    analyzer = TicTacToeMoveAnalyzer(moves, rewards, 3)
    assert analyzer.moves_won == moves_won[-3:]
    analyzer = TicTacToeMoveAnalyzer(moves[:-1], rewards[:-1], 5)
    assert analyzer.moves_won == moves_won[-4:-1]

def test_constructor_out_of_bound(moves, rewards):
    # test with the boarderline within bound
    TicTacToeMoveAnalyzer(moves, rewards, 0)
    TicTacToeMoveAnalyzer(moves, rewards, 11)
    with pytest.raises(IndexError):
        TicTacToeMoveAnalyzer(moves, rewards, 12)

def test_num_wins(moves, rewards):
    # wins [0, 2, 3, 4, 6, 8, 9, 10]
    analyzer = TicTacToeMoveAnalyzer(moves, rewards, 10)
    assert analyzer.num_wins() == 7
    analyzer = TicTacToeMoveAnalyzer(moves, rewards, 5)
    assert analyzer.num_wins() == 4
    analyzer = TicTacToeMoveAnalyzer(moves[:-1], rewards[:-1], 5)
    assert analyzer.num_wins() == 3

def test_num_wins_with_corner_opening(moves, rewards):
    # wins with corner opening [0, 3, 4, 6, 9, 10]
    analyzer = TicTacToeMoveAnalyzer(moves, rewards, 10)
    assert analyzer.num_wins_with_corner_opening() == 5 # [3, 4, 6, 9, 10]
    analyzer = TicTacToeMoveAnalyzer(moves, rewards, 3)
    assert analyzer.num_wins_with_corner_opening() == 2 # [9, 10]

def test_num_wins_with_corner_opening_and_center(moves, rewards):
    # wins with corner opening and then center [0, 4, 6, 10]
    analyzer = TicTacToeMoveAnalyzer(moves, rewards, 10)
    assert analyzer.num_wins_with_corner_opening_and_center() == 3 # [4, 6, 10]
    analyzer = TicTacToeMoveAnalyzer(moves, rewards, 3)
    assert analyzer.num_wins_with_corner_opening_and_center() == 1 # [10]

def test_opening_moves(moves, rewards):
    # opening moves [0, 0, 4, 0, 2, 4, 0, 0, 4, 0, 2]
    analyzer = TicTacToeMoveAnalyzer(moves, rewards, 10)
    assert analyzer.opening_moves == [0, 4, 0, 2, 4, 0, 0, 4, 0, 2]
    analyzer = TicTacToeMoveAnalyzer(moves, rewards, 3)
    assert analyzer.opening_moves == [4, 0, 2]
    analyzer = TicTacToeMoveAnalyzer(moves[:-1], rewards[:-1], 5)
    assert analyzer.opening_moves == [4, 0, 0, 4, 0]

def test_num_corner_opening_games(moves, rewards):
    # corner opening games [0, 1, 3, 4, 6, 7, 9, 10]
    analyzer = TicTacToeMoveAnalyzer(moves, rewards, 10)
    assert analyzer.num_corner_openings() == 7
    analyzer = TicTacToeMoveAnalyzer(moves, rewards, 3)
    assert analyzer.num_corner_openings() == 2
    analyzer = TicTacToeMoveAnalyzer(moves[:-1], rewards[:-1], 5)
    assert analyzer.num_corner_openings() == 3

