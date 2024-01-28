import os
import random
import numpy as np
import tqdm
from game import Game, MonteCarloPlayerMB,MonteCarloPlayer_classic,MonteCarloPNSPlayer, Move, Player
from board import Board
from tree import MonteCarloTreeSearchNode
from tree_MB import MonteCarloTreeSearchNodeMB
from time import time



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
    tot = 50

    # cross validation backbone to find best hyperparameters
        # this below is the best configuration found if we consider a performance/execution_time tradeoff
    duration = 1 #in terms of seconds
    for ns in [1000]:
        for cp in [0.1]:

            #wins and matches for accuracy
            wins = 0
            matches = 0

            #play tot games
            for i in tqdm.tqdm(range(tot)):
                my_player_id = random.randint(0, 1)
                opposer = 1 - my_player_id
                print(f"\nmy_player_id: {my_player_id}")
                g = Game()
                #g.print()

                # player initialization -> our player is players[my_player_id]
                minmax_depth = 1
                MR_hybrid = True # if True, the player will use the Minimax hybrid algorithm for the rollout

                #NB: sono passato dall'usare una duration a tornare al numero di iterazioni massimo
                players[my_player_id] = MonteCarloPNSPlayer(player_id=my_player_id,duration=duration, c_param=0.5, pn_param=0.5,MR_hybrid = MR_hybrid,minimax_depth=minmax_depth)

                root_classic = MonteCarloTreeSearchNode(state=Board(), player_id=opposer, d=0, id=0,root_player=opposer, num_simulations=ns,c_param=cp)
                players[opposer] = MonteCarloPlayer_classic(root=root_classic, player_id=opposer,num_simulations=ns, c_param=cp,duration = duration)
                #players[opposer] = RandomPlayer()
                
                # play the game
                winner = g.play(players[0], players[1])
                g.print()
                print(f"Winner: Player {winner}")
                matches += 1

                #update accuracy
                if winner == my_player_id:
                    print("My player WON!")
                    wins += 1
                else:
                    #g.print()
                    print("My player LOST!")
                    losing_board.append(g.get_board())
                acc  = 100*float(wins)/float(matches)
                print(f"Winning rate: {acc}% with {wins} wins on {matches} matches" )

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

