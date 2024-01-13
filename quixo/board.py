from enum import Enum
import numpy as np
import itertools
import random


class Move(Enum):
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3

class Board():
    def __init__(self,board=None):
        if board is not None:
            self._board = board
        else:
            self._board = np.ones((5, 5), dtype=np.uint8) * -1

    # Metodo privato al nodo per sapere tutti i possibili movimenti di slide concessi  datta posizione from_position
    def __acceptable_slides(self, from_position: tuple[int, int]):
        """When taking a piece from {from_position} returns the possible moves (slides)"""
        acceptable_slides = [Move.BOTTOM, Move.TOP, Move.LEFT, Move.RIGHT]
        axis_1 = from_position[0]    # axis_0 = 0 means uppermost row
        axis_0 = from_position[1]    # axis_1 = 0 means leftmost column

        if axis_0 == 0:  # can't move upwards if in the top row...
            acceptable_slides.remove(Move.TOP)
        elif axis_0 == 4:
            acceptable_slides.remove(Move.BOTTOM)

        if axis_1 == 0:
            acceptable_slides.remove(Move.LEFT)
        elif axis_1 == 4:
            acceptable_slides.remove(Move.RIGHT)
        return acceptable_slides
    

    def get_legal_actions(self, player_id):
        from_all_possible_pos = list(itertools.product(list(range(5)), repeat=2))
        possible_moves = []
        for from_pos in from_all_possible_pos:
            row, col = from_pos
            from_pos = tuple([col, row])
            from_border = row in (0, 4) or col in (0, 4)
            if not from_border:
                continue  # the cell is not in the border
            if self._board[from_pos] != player_id and self._board[from_pos] != -1:
                continue  # the cell belongs to the opponent
            accettable_slides = self.__acceptable_slides(tuple([row,col]))
            for slide in accettable_slides:
                possible_moves.append((tuple([row,col]), slide))
        # I shuffle the possible moves in order to use the pop function in the expand method
        random.shuffle(possible_moves)
        return possible_moves
    

    # metodo per applicare la mossa 
    def move(self, action, p_id):
        from_pos, slide = action
        # _slide
        axis_1, axis_0 = from_pos
        # np.roll performs a rotation of the element of a 1D ndarray
        if slide == Move.RIGHT:
            self._board[axis_0, axis_1:] = np.roll(self._board[axis_0, axis_1:], -1)
            self._board[axis_0, 4] = p_id
        elif slide == Move.LEFT:
            self._board[axis_0, 0:axis_1+1] = np.roll(self._board[axis_0, 0:axis_1+1], 1)
            self._board[axis_0, 0] = p_id
        elif slide == Move.BOTTOM:
            self._board[axis_0:, axis_1] = np.roll(self._board[axis_0:, axis_1], -1)
            self._board[4, axis_1] = p_id
        elif slide == Move.TOP:
            self._board[:axis_0+1, axis_1] = np.roll(self._board[:axis_0+1, axis_1], 1)
            self._board[0, axis_1] = p_id


    # Sembra non tenga in considerazione il pareggio
    def check_winner(self) -> int:
        '''Check the winner. Returns the player ID of the winner if any, otherwise returns -1'''
        # for each row
        for x in range(self._board.shape[0]):
            # if a player has completed an entire row
            if self._board[x, 0] != -1 and all(self._board[x, :] == self._board[x, 0]):
                # return the relative id
                return self._board[x, 0]
        # for each column
        for y in range(self._board.shape[1]):
            # if a player has completed an entire column
            if self._board[0, y] != -1 and all(self._board[:, y] == self._board[0, y]):
                # return the relative id
                return self._board[0, y]
        # if a player has completed the principal diagonal
        if self._board[0, 0] != -1 and all(
            [self._board[x, x]
                for x in range(self._board.shape[0])] == self._board[0, 0]
        ):
            # return the relative id
            return self._board[0, 0]
        # if a player has completed the secondary diagonal
        if self._board[0, -1] != -1 and all(
            [self._board[x, -(x + 1)]
             for x in range(self._board.shape[0])] == self._board[0, -1]
        ):
            # return the relative id
            return self._board[0, -1]
        return -1
    
    def printami(self):
        print(self._board)