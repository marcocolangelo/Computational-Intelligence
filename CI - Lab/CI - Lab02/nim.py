import logging
from collections import namedtuple
import numpy as np

Nimply = namedtuple("Nimply", "row, num_objects")
Results = namedtuple("Results", "winner, turns")

class Nim:
    def __init__(self, num_rows: int, k: int = None) -> None:
        self._rows = [i * 2 + 1 for i in range(num_rows)]
        self._k = k

    def __bool__(self):
        return sum(self._rows) > 0

    def __str__(self):
        return "<" + " ".join(str(_) for _ in self._rows) + ">"

    @property
    def rows(self) -> tuple:
        return tuple(self._rows)

    def nimming(self, ply: Nimply) -> None:
        row, num_objects = ply
        assert self._rows[row] >= num_objects
        assert self._k is None or num_objects <= self._k
        self._rows[row] -= num_objects


def play_game(strategy): 
    logging.getLogger().setLevel(logging.INFO) 
 
    nim = Nim(5) 
    #logging.info(f"init : {nim}") 
    player = 0 
 
    # 'turns' count the number of turns the match last (It's a good metric to use for the fitness) 
    turns = 0 
 
    # number of sticks moved by oppo in the entire game 
    #sticks_oppo = 0 
    while nim: 
        ply = strategy[player](nim) 
        #sticks_oppo += ply.num_objects 
       # logging.info(f"ply: player {player} plays {ply}") 
        nim.nimming(ply) 
        #logging.info(f"status: {nim}") 
        # A new turn starts when the playes has done the first move plays again 
        if player == 0: 
            turns += 1 
        player = 1 - player 
    #logging.info(f"status: Player {player} won!") 
 
    return Results(player, turns)

def play_game2(strategy): 
    #logging.getLogger().setLevel(logging.INFO) 
 
    nim = Nim(5) 
    #logging.info(f"init : {nim}") 
    player = 0 
 
    # 'turns' count the number of turns the match last (It's a good metric to use for the fitness) 
    turns = 0 
 
    # number of sticks moved by oppo in the entire game 
    #sticks_oppo = 0 
    while nim: 
        ply = strategy[player](nim) 
        #sticks_oppo += ply.num_objects 
        #logging.info(f"ply: player {strategy[player]} plays {ply}") 
        nim.nimming(ply) 
        #logging.info(f"status: {nim}") 
        # A new turn starts when the playes has done the first move plays again 
        if player == 0: 
            turns += 1 
        player = 1 - player 
    #logging.info(f"status: Player {strategy[player]} won!") 
 
    return player
