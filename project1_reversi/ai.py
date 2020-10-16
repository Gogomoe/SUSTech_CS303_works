import random
from collections import deque
from typing import List, Tuple

import numpy as np

INFINITY = 1000000000

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)

# @formatter:off
point_weight = np.array([
    [100, -10,  8,  6,  6,  8, -10, 100],
    [-10, -25, -4, -4, -4, -4, -25, -10],
    [  8,  -4,  6,  4,  4,  6,  -4,   8],
    [  6,  -4,  4,  0,  0,  4,  -4,   6],
    [  6,  -4,  4,  0,  0,  4,  -4,   6],
    [  8,  -4,  6,  4,  4,  6,  -4,   8],
    [-10, -25, -4, -4, -4, -4, -25, -10],
    [100, -10,  8,  6,  6,  8, -10, 100]
])
# @formatter:on


class Evaluator:

    def __init__(self, ai):
        self.ai = ai
        self.chessboard_size = ai.chessboard_size
        self.color = ai.color

    def evaluate(self, chessboard: np.ndarray, current_player: int) -> int:

        result = 0

        weight = self.weights_game_phase(chessboard)

        result += weight[0] * self.placement(chessboard)
        result += weight[1] * self.pieces(chessboard)
        result += weight[2] * self.mobility(chessboard)
        result += weight[3] * self.stability(chessboard)
        result += weight[4] * self.corner(chessboard, current_player)

        return result

    def weights_game_phase(self, chessboard: np.ndarray) -> List[int]:
        size = self.chessboard_size
        step = 0

        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                step += abs(chessboard[i, j])

        if step / size / size < 0.4:
            return [
                10,  # 0 placement
                10,  # 1 pieces
                50,  # 2 mobility
                200,  # 3 steady
                1000,  # 4 corner
            ]
        elif step / size / size < 0.7:
            return [
                10,  # 0 placement
                10,  # 1 pieces
                20,  # 2 mobility
                200,  # 3 steady
                1000,  # 4 corner
            ]
        else:
            return [
                5,  # 0 placement
                50,  # 1 pieces
                50,  # 2 mobility
                200,  # 3 steady
                1000,  # 4 corner
            ]

    def pieces(self, chessboard: np.ndarray) -> int:
        my_piece = 0
        op_piece = 0
        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                if chessboard[i][j] == self.color:
                    my_piece += 1
                if chessboard[i][j] == -self.color:
                    op_piece += 1
        return 100 * (my_piece - op_piece) // (my_piece + op_piece + 1)

    def placement(self, chessboard: np.ndarray) -> int:
        weight: np.ndarray
        if self.chessboard_size == 8:
            weight = point_weight
        else:
            weight = np.ones((self.chessboard_size, self.chessboard_size))

        my_weight = 0
        op_weight = 0
        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                if chessboard[i][j] == self.color:
                    my_weight += weight[i][j]
                if chessboard[i][j] == -self.color:
                    op_weight += weight[i][j]
        return my_weight - op_weight

    def mobility(self, chessboard: np.ndarray) -> int:
        my_mobility = len(self.ai.actions(chessboard, self.color))
        op_mobility = len(self.ai.actions(chessboard, -self.color))
        return 100 * (my_mobility - op_mobility) // (my_mobility + op_mobility + 1)

    def stability(self, chessboard: np.ndarray) -> int:
        my_stability = 0
        op_stability = 0

        # 0 -> not search, -1 -> black, 1 -> white, 2 -> not steady, 3 -> out bound
        size = self.chessboard_size
        steady_board = np.zeros((size, size))
        point_queue = deque([(0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1)])

        def check_steady(y, x):
            if x < 0 or x >= size or y < 0 or y >= size:
                return 3
            if abs(steady_board[y, x]) == 1:
                return steady_board[y, x]
            return 2

        while point_queue:
            y, x = point_queue.popleft()
            if x < 0 or x >= size or y < 0 or y >= size:
                continue
            if steady_board[y, x] != 0:
                continue

            color = chessboard[y, x]
            if color == 0:
                steady_board[y, x] = 2
                continue
            for dy, dx in [(0, 1), (1, 1), (1, 0), (1, -1)]:
                dir1 = check_steady(y + dy, x + dx)
                dir2 = check_steady(y - dy, x - dx)

                one_side_empty = dir1 == 3 or dir2 == 3
                one_side_steady = dir1 == color or dir2 == color
                two_side_enemy_steady = dir1 == -color and dir2 == -color
                if not (one_side_empty or one_side_steady or two_side_enemy_steady):
                    steady_board[y, x] = 2
                    break
            if steady_board[y, x] == 0:
                steady_board[y, x] = color
                if color == self.color:
                    my_stability += 1
                if color == -self.color:
                    op_stability += 1
            point_queue.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])

        return 100 * (my_stability - op_stability) // (my_stability + op_stability + 1)

    def corner(self, chessboard: np.ndarray, current_player: int) -> int:
        actions = self.ai.actions(chessboard, current_player)
        last = self.chessboard_size - 1
        for action in actions:
            if action == (0, 0) or action == (0, last) or action == (last, 0) or action == (last, last):
                return 100 * (1 if current_player == self.color else -1)
        return 0


# don't change the class name
class AI(object):

    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size: int, color: int, time_out: int):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.evaluator = Evaluator(self)

    # The input is current chessboard.
    def go(self, chessboard: np.ndarray):
        self.candidate_list.clear()

        actions = self.actions(chessboard, self.color)

        self.candidate_list.extend(actions)
        if len(actions) <= 1:
            return

        depth = 4

        # final_step = 10
        # step = 0
        # for i in range(self.chessboard_size):
        #     for j in range(self.chessboard_size):
        #         step += abs(chessboard[i, j])
        # space = self.chessboard_size * self.chessboard_size - step
        # if space <= final_step:
        #     depth = final_step - 1

        best_score = -INFINITY

        greedy_corner = False
        last = self.chessboard_size - 1
        for action in actions:
            if action == (0, 0) or action == (0, last) or action == (last, 0) or action == (last, last):
                new_chessboard = self.make_move(chessboard, action, self.color)
                v = self.evaluate(new_chessboard, -self.color)
                if v > best_score:
                    best_score = v
                    greedy_corner = True
                    self.candidate_list.append(action)
        if greedy_corner:
            return

        best_score = -INFINITY
        beta = INFINITY

        for action in actions:
            new_chessboard = self.make_move(chessboard, action, self.color)
            v = self.min_value(new_chessboard, best_score, beta, depth)
            if v > best_score:
                best_score = v
                self.candidate_list.append(action)

    def actions(self, chessboard: np.ndarray, color: int) -> List[Tuple[int, int]]:
        def check_point(sy: int, sx: int) -> bool:
            if chessboard[sy, sx] != 0:
                return False
            for direct in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                y = sy + direct[0]
                x = sx + direct[1]
                if 0 <= x < self.chessboard_size and 0 <= y < self.chessboard_size and chessboard[y, x] != -color:
                    continue
                while 0 <= x < self.chessboard_size and 0 <= y < self.chessboard_size and chessboard[y, x] == -color:
                    y += direct[0]
                    x += direct[1]
                if 0 <= x < self.chessboard_size and 0 <= y < self.chessboard_size and chessboard[y, x] == color:
                    return True
            return False

        result = []
        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                if check_point(i, j):
                    result.append((i, j))

        return result

    def max_value(self, chessboard: np.ndarray, alpha: int, beta: int, depth: int) -> int:
        current_player = self.color
        if depth <= 0:
            return self.evaluate(chessboard, current_player)
        actions = self.actions(chessboard, current_player)
        if len(actions) == 0:
            return self.min_value(chessboard, alpha, beta, depth - 1)
        v = -INFINITY
        for move in actions:
            new_chessboard = self.make_move(chessboard, move, current_player)
            v = max(v, self.min_value(new_chessboard, alpha, beta, depth - 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, chessboard: np.ndarray, alpha: int, beta: int, depth: int) -> int:
        current_player = -self.color
        if depth <= 0:
            return self.evaluate(chessboard, current_player)
        actions = self.actions(chessboard, current_player)
        if len(actions) == 0:
            return self.max_value(chessboard, alpha, beta, depth - 1)
        v = INFINITY
        for move in actions:
            new_chessboard = self.make_move(chessboard, move, current_player)
            v = min(v, self.max_value(new_chessboard, alpha, beta, depth - 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def evaluate(self, chessboard: np.ndarray, current_player: int) -> int:
        return self.evaluator.evaluate(chessboard, current_player)

    def make_move(self, chessboard: np.ndarray, move: Tuple[int, int], color: int):
        new = chessboard.copy()
        sy, sx = move
        for direct in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            y = sy + direct[0]
            x = sx + direct[1]
            if 0 <= x < self.chessboard_size and 0 <= y < self.chessboard_size and new[y, x] != -color:
                continue
            while 0 <= x < self.chessboard_size and 0 <= y < self.chessboard_size and new[y, x] == -color:
                y += direct[0]
                x += direct[1]
            if 0 <= x < self.chessboard_size and 0 <= y < self.chessboard_size and new[y, x] == color:
                while y != sy or x != sx:
                    y -= direct[0]
                    x -= direct[1]
                    new[y, x] = color
        return new
