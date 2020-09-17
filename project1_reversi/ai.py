import numpy as np
import random
import time

from typing import List, Tuple

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)


# don't change the class name
class AI(object):

    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size: int, color: int, time_out: int):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need add your decision into your candidate_list.
        # System will get the end of your candidate_list as your decision.
        self.candidate_list = []

    # The input is current chessboard.
    def go(self, chessboard: np.ndarray):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        # ==================================================================
        # Write your algorithm here
        # Here is the simplest sample:Random decision

        actions = self.actions(chessboard, self.color)

        self.candidate_list.extend(actions)
        if len(actions) != 0:
            self.candidate_list.append(random.choice(actions))

        # ==============Find new pos========================================
        # Make sure that the position of your decision in chess board is empty.
        # If not, the system will return error.
        # Add your decision into candidate_list, Records the chess board
        # You need add all the positions which is valid
        # candidate_list example: [(3,3),(4,4)]
        # You need append your decision at the end of the candidate_list,
        # we will choose the last element of the candidate_list as the position you choose
        # If there is no valid position, you must return a empty list.

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
