
class TicTacToeMoveAnalyzer:
    def __init__(self, moves, rewards, num_last_games_to_analyze):
        if num_last_games_to_analyze < 0:
            raise ValueError
        self.opening_moves = [m[0] for m in moves[-num_last_games_to_analyze:]]
        self.moves_won = [moves[i] for i in range(-num_last_games_to_analyze, 0) if rewards[i] == 1]

    def num_wins(self):
        return len(self.moves_won)

    def num_corner_openings(self):
        corners = [0, 2, 6, 8]
        return len([1 for om in self.opening_moves if om in corners])

    def num_wins_with_corner_opening(self):
        corners = [0, 2, 6, 8]
        return len([1 for m in self.moves_won if m[0] in corners])

    def num_wins_with_corner_opening_and_center(self):
        corners = [0, 2, 6, 8]
        return len([1 for m in self.moves_won if m[0] in corners and m[2] == 4])

    