import os
import random
import numpy as np
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
    
    losing_board = []
    results = {}
    my_player_id = 0
    players = np.empty(2, dtype=Player)
    tot = 10

    # cross validation backbone to find best hyperparameters
        # this below is the best configuration found if we consider a performance/execution_time tradeoff
    for ns in [100]:
        for cp in [0.1]:

            #wins and matches for accuracy
            wins = 0
            matches = 0

            #play tot games
            for i in tqdm.tqdm(range(tot)):
                my_player_id = random.randint(0, 1)
                print(f"my_player_id: {my_player_id}")
                g = Game()
                g.print()

                # player initialization -> our player is players[my_player_id]
                root = MonteCarloTreeSearchNode(state=Board(), player_id=my_player_id, d=0, id=0,root_player=my_player_id, num_simulations=ns,c_param=cp)
                players[my_player_id] = MonteCarloPlayer(root=root, player_id=my_player_id,num_simulations=ns, c_param=cp)
                players[1 - my_player_id] = RandomPlayer()
                
                # play the game
                winner = g.play(players[0], players[1])
                g.print()
                print(f"Winner: Player {winner}")
                matches += 1

                #update accuracy
                if winner == my_player_id:
                    wins += 1
                else:
                    #g.print()
                    losing_board.append(g.get_board())

            #print accuracy
            acc  = 100*float(wins)/float(matches)
            print("accuracy: ",acc )

            #save results
            results[str(ns) + "-"+ str(cp)] = acc
   
            # with open('results.txt', 'a' if os.path.isfile('results.txt') else 'w') as file:
            #             file.write("\nConfig: "+ str(str(ns) + "-"+ str(cp)))
            #             file.write("\n")
            #             file.write(str(results))    
            #             print(results)

            # with open('lost.txt', 'a' if os.path.isfile('lost.txt') else 'w') as file:
            #             file.write("\nConfig: "+ str(str(ns) + "-"+ str(cp)))
            #             file.write("\n")
            #             file.write(str(losing_board))    
            #             print(losing_board)
    
    # MonteCarloTreeSearchNode.main()

