import numpy as np
import random
import time
from collections import deque

from typing import List, Tuple

INFINITY = 1000000000

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)


# don't change the class name
class AI(object):

    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size: int, color: int, time_out: int):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []

    # The input is current chessboard.
    def go(self, chessboard: np.ndarray):
        self.candidate_list.clear()

        actions = self.actions(chessboard, self.color)

        self.candidate_list.extend(actions)
        if len(actions) <= 1:
            return

        depth = 4

        result_max = -INFINITY

        for action in actions:
            new_chessboard = self.make_move(chessboard, action, self.color)
            result = -self.alpha_beta(new_chessboard, -INFINITY, INFINITY, -self.color, depth)
            if result > result_max:
                result_max = result
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

    def alpha_beta(self, chessboard: np.ndarray, alpha: int, beta: int, color: int, depth: int) -> int:
        sign = {self.color: 1, -self.color: -1}
        result_max = -INFINITY
        if depth <= 0:
            return sign[color] * self.evaluate(chessboard)
        if not self.can_move(chessboard, color):
            if not self.can_move(chessboard, -color):
                return sign[color] * self.evaluate(chessboard)
            return -self.alpha_beta(chessboard, -beta, -alpha, -color, depth)

        for move in self.actions(chessboard, color):
            new_chessboard = self.make_move(chessboard, move, color)
            val = -self.alpha_beta(new_chessboard, -beta, -alpha, -color, depth - 1)
            if val > alpha:
                if val >= beta:
                    return val
                alpha = max(alpha, val)
            result_max = max(result_max, val)
        return result_max

    def evaluate(self, chessboard: np.ndarray) -> int:
        weight = [
            6,  # 0 corner
            -3,  # 1 corner danger
            -6,  # 2 more danger
            2,  # 3 point
            2,  # 4 mobility
            4,  # 5 steady
        ]

        result = 0
        sign = {self.color: 1, -self.color: -1, 0: 0}

        for (i, j) in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
            result += sign[chessboard[i, j]] * weight[0]

        for (i, j) in [(0, 1), (1, 0), (0, -2), (-2, 0), (1, -1), (-1, 1), (-1, -2), (-2, -1)]:
            result += sign[chessboard[i, j]] * weight[1]

        for (i, j) in [(1, 1), (1, -2), (-2, 1), (-2, -2)]:
            result += sign[chessboard[i, j]] * weight[2]

        size = self.chessboard_size
        for i in range(size):
            for j in range(size):
                result += sign[chessboard[i, j]] * weight[3]

        result += len(self.actions(chessboard, self.color)) * weight[4]

        # 0 -> not search, -1 -> black, 1 -> white, 2 -> not steady, 3 -> out bound
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
                result += sign[color] * weight[5]
            point_queue.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])

        return result

    def can_move(self, chessboard: np.ndarray, color: int) -> bool:
        return len(self.actions(chessboard, color)) > 0

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
