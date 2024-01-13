from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
import numpy as np
from board import Board
from tree import MonteCarloTreeSearchNode

# Rules on PDF
class Player():
    def __init__(self):
        self.p = 1

class Move(Enum):
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3

## MonteCarlo Fist Visit approach
class MonteCarloPlayer(Player):
    def __init__(self,root : MonteCarloTreeSearchNode, num_simulations = 100, c_param = 0.1) -> None:
            self.root = root
            self.num_simulations = num_simulations
            self.c_param = c_param
            

    @abstractmethod
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        root = MonteCarloTreeSearchNode(Board(game.get_board()), 0, 0, 0,num_simulations=self.num_simulations, c_param=self.c_param)
        selected_node = root.best_action()
        from_pos, move = selected_node.parent_action
        #print('In make_move del nostro player -> Il nodo selezionato è il seguente: ', selected_node)
        #print(f"In make_move del nostro player -> from_pos: {from_pos}, move: {move}")
        return from_pos, Move(move.value)
 
        


class Game(object):
    def __init__(self) -> None:
        self._board = np.ones((5, 5), dtype=np.uint8) * -1

    def get_board(self):
        '''
        Returns the board
        '''
        return deepcopy(self._board)

    def print(self):
        '''Prints the board. -1 are neutral pieces, 0 are pieces of player 0, 1 pieces of player 1'''
        print(self._board)

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

    def play(self, player1: Player, player2: Player) -> int:
        '''Play the game. Returns the winning player'''
        players = [player1, player2]
        current_player_idx = 1
        winner = -1
        while winner < 0:
            current_player_idx += 1
            current_player_idx %= len(players)
            ok = False
            #print(f"Nuovo turno : player {current_player_idx}")
            while not ok:
                from_pos, slide = players[current_player_idx].make_move(self)
                ok = self.__move(from_pos, slide, current_player_idx)
                #print(f"In play -> ok: {ok}")
            # self.print()
            winner = self.check_winner()
        return winner

## private function, so it is not visible from the outside
    def __move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        '''Perform a move'''
        if player_id > 2:
           # print(f"In __move -> player_id: {player_id} quindi ritorno False")
            return False
        # Oh God, Numpy arrays
        prev_value = deepcopy(self._board[(from_pos[1], from_pos[0])])
        acceptable = self.__take((from_pos[1], from_pos[0]), player_id)
        # print(f"In __move ->  primo acceptable: {acceptable}")
        if acceptable:
            acceptable = self.__slide((from_pos[1], from_pos[0]), slide, player_id)
            # print(f"In __move -> secondo acceptable: {acceptable} perchè __slide ritorna {acceptable}")
            if not acceptable:
                self._board[(from_pos[1], from_pos[0])] = deepcopy(prev_value)
        return acceptable

    def __take(self, from_pos: tuple[int, int], player_id: int) -> bool:
        """Checks that {from_pos} is in the border and marks the cell with {player_id}"""
        row, col = from_pos
        from_border = row in (0, 4) or col in (0, 4)
        if not from_border:
            # print(f"    In __take ->  from_border: {from_border} quindi ritorno False")
            return False  # the cell is not in the border
        if self._board[from_pos] != player_id and self._board[from_pos] != -1:
            # print(f"    In __take ->  self._board[from_pos]: {self._board[from_pos]} quindi ritorno False")
            return False  # the cell belongs to the opponent
        #self._board[from_pos] = player_id
        return True
    

    def __acceptable_slides(self, from_position: tuple[int, int]):
        """When taking a piece from {from_position} returns the possible moves (slides)"""
        acceptable_slides = list([Move.BOTTOM, Move.TOP, Move.LEFT, Move.RIGHT])
        

        axis_0 = from_position[0]    # axis_0 = 0 means uppermost row
        axis_1 = from_position[1]    # axis_1 = 0 means leftmost column

        if axis_0 == 0:  # can't move upwards if in the top row...
            # print(f"        In __acceptable_slides ->  axis_0: {axis_0} quindi rimuovo TOP")
            acceptable_slides.remove(Move.TOP)
        elif axis_0 == 4:
            # print(f"        In __acceptable_slides ->  axis_0: {axis_0} quindi rimuovo BOTTOM")
            acceptable_slides.remove(Move.BOTTOM)

        if axis_1 == 0:
            # print(f"        In __acceptable_slides ->  axis_1: {axis_1} quindi rimuovo LEFT")
            acceptable_slides.remove(Move.LEFT)
        elif axis_1 == 4:
            # print(f"        In __acceptable_slides ->  axis_1: {axis_1} quindi rimuovo RIGHT")
            acceptable_slides.remove(Move.RIGHT)
        
        # print(f"        In __acceptable_slides ->  acceptable_slides: {acceptable_slides}")
        return acceptable_slides
    

    def __slide(self, from_pos: tuple[int, int], slide: Move, p_id) -> bool:
        '''Slide the other pieces'''
        acc_slide = self.__acceptable_slides(from_pos)
        if slide not in acc_slide:
            # print(f"    In__slide -> slide: {slide} not in acceptable slides :{acc_slide}")
            return False  # consider raise ValueError('Invalid argument value')
        axis_0, axis_1 = from_pos
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
        return True
