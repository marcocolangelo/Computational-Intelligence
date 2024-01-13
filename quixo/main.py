import random
from game import Game, MonteCarloPlayer, Move, Player
from board import Board
from tree import MonteCarloTreeSearchNode


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


# class MyPlayer(Player):
#     def __init__(self) -> None:
#         super().__init__()

#     def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
#         from_pos = (random.randint(0, 4), random.randint(0, 4))
#         move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
#         return from_pos, move


if __name__ == '__main__':
    g = Game()
    g.print()
    root = MonteCarloTreeSearchNode(Board(), 0, 0, 0, num_simulations=10)
    player1 = MonteCarloPlayer(root)
    player2 = RandomPlayer()
    winner = g.play(player1, player2)
    g.print()
    print(f"Winner: Player {winner}")
    # MonteCarloTreeSearchNode.main()

