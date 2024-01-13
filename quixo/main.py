import os
import random
import tqdm
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
    
    ns = 100
    wins = 0
    matches = 0
    losing_board = []
    results = {}
    for ns in [1, 10,100,1000]:
        for cp in [0.1,0.5,0.9]:
            wins = 0
            matches = 0
            for i in tqdm.tqdm(range(100)):
                g = Game()
                #g.print()
                root = MonteCarloTreeSearchNode(Board(), 0, 0, 0, num_simulations=ns,c_param=cp)
                player1 = MonteCarloPlayer(root, num_simulations=ns, c_param=cp)
                player2 = RandomPlayer()
                winner = g.play(player1, player2)
                #g.print()
                print(f"Winner: Player {winner}")
                matches += 1
                if winner == 0:
                    wins += 1
                else:
                    g.print()
                    losing_board.append(g.get_board())

            acc  = 100*float(wins)/float(matches)
            print("accuracy: ",acc )

            results[str(ns) + "-"+ str(cp)] = acc
   
            with open('results.txt', 'a' if os.path.isfile('results.txt') else 'w') as file:
                        file.write("\nConfig: "+ str(str(ns) + "-"+ str(cp)))
                        file.write("\n")
                        file.write(str(results))    
                        print(results)

            with open('lost.txt', 'a' if os.path.isfile('lost.txt') else 'w') as file:
                        file.write("\nConfig: "+ str(str(ns) + "-"+ str(cp)))
                        file.write("\n")
                        file.write(str(losing_board))    
                        print(losing_board)
    
    # MonteCarloTreeSearchNode.main()

