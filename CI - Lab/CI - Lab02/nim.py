import logging
from collections import namedtuple
import numpy as np

Nimply = namedtuple("Nimply", "row, num_objects")

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

    #strategy = (optimal, pure_random)

    nim = Nim(5)
    logging.info(f"init : {nim}")
    player = 0
    while nim:
        ply = strategy[player](nim)
        logging.info(f"ply: player {player} plays {ply}")
        nim.nimming(ply)
        logging.info(f"status: {nim}")
        player = 1 - player
    logging.info(f"status: Player {player} won!")
