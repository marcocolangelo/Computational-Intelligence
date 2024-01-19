from abc import abstractmethod
import os
import random
import numpy as np
import tqdm
from game import Game, MonteCarloPlayer, Move, Player
from board import Board
from tree import MonteCarloTreeSearchNode
from tree_copy import MonteCarloTreeSearchNodeNoGreedy

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

## MonteCarlo Fist Visit approach
class MonteCarloPlayer_classic(Player):
    def __init__(self,root : MonteCarloTreeSearchNodeNoGreedy,player_id, num_simulations = 100, c_param = 0.1) -> None:
            self.root = root
            self.num_simulations = num_simulations
            self.c_param = c_param
            self.player_id = player_id

            

    @abstractmethod
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        #print(f"make_move -> my player id: {self.player_id}")
        root = MonteCarloTreeSearchNodeNoGreedy(Board(game.get_board()), player_id=self.player_id, d=0, root_player=self.player_id,id=0,num_simulations=self.num_simulations, c_param=self.c_param)
        selected_node = root.best_action()
        from_pos, move = selected_node.parent_action
        #print('In make_move del nostro player -> Il nodo selezionato è il seguente: ', selected_node)
        #print(f"In make_move del nostro player -> from_pos (col,row): {from_pos}, move: {move}")
        return from_pos, Move(move.value)

if __name__ == '__main__':
    
    losing_board = []
    results = {}
    my_player_id = 0
    players = np.empty(2, dtype=Player)
    tot = 100

    # cross validation backbone to find best hyperparameters
        # this below is the best configuration found if we consider a performance/execution_time tradeoff
    for ns in [100]:
        for cp in [0.1]:
            for cp2 in [0.1]:
                #wins and matches for accuracy
                wins = 0
                matches = 0

                #play tot games
                for i in tqdm.tqdm(range(tot)):
                    my_player_id = random.randint(0, 1)
                    print(f"my_player_id: {my_player_id}")
                    g = Game()
                    #g.print()

                    # player initialization -> our player is players[my_player_id]
                    root = MonteCarloTreeSearchNode(state=Board(), player_id=my_player_id, d=0, id=0,root_player=my_player_id, num_simulations=ns,c_param=cp)
                    players[my_player_id] = MonteCarloPlayer_classic(root=root, player_id=my_player_id,num_simulations=ns, c_param=cp)
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
                    print(f"Vinte {wins} partite su {matches} : cp :{cp} e cp2:{cp2} accuracy = {100*float(wins)/float(matches)}%")

                #print accuracy
                acc  = 100*float(wins)/float(matches)
                print(f"cp :{cp} e cp2:{cp2} ->  accuracy: ",acc )

                #save results
                results[str(ns) + "-"+ str(cp)] = acc
    

