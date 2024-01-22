import os
import random
import numpy as np
import tqdm
from game import Game, Move, Player
from board import Board
from tree import MonteCarloTreeSearchNode, PN_MCTS_Node



class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


## MonteCarlo Tree Search 
class MonteCarloPlayer(Player):
    def __init__(self, player_id, num_simulations = 100, c_param = 0.1) -> None:
            self.num_simulations = num_simulations
            self.c_param = c_param
            self.player_id = player_id
            

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        root = MonteCarloTreeSearchNode(Board(game.get_board()), player_id = self.player_id, 
                                        root_player = self.player_id, num_simulations = self.num_simulations, 
                                        c_param = self.c_param)
        selected_node = root.best_action()
        from_pos, move = selected_node.get_action()
        #print('In make_move del nostro player -> Il nodo selezionato è il seguente: ', selected_node)
        #print(f"In make_move del nostro player -> from_pos (col,row): {from_pos}, move: {move}")
        return from_pos, move
    

# Player che implementa solver
class MonteCarloPNSPlayer(Player):
    def __init__(self, player_id, duration = 1, c_param = 0.1, pn_param = 0.1) -> None:
        self.duration = duration
        self.c_param = c_param
        self.pn_param = pn_param
        self.player_id = player_id

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        root = PN_MCTS_Node(Board(game.get_board()), player_id = self.player_id, 
                                        root_player = self.player_id, duration = self.duration, 
                                        c_param = self.c_param, pn_param = self.pn_param)
        selected_node = root.best_action()
        from_pos, move = selected_node.get_action()
        #print('In make_move del nostro player -> Il nodo selezionato è il seguente: ', selected_node)
        #print(f"In make_move del nostro player -> from_pos (col,row): {from_pos}, move: {move}")
        return from_pos, move
    

## MonteCarlo First Visit approach
# class MonteCarloPlayer_minimax(Player):
#     def __init__(self,root : MonteCarloTreeSearchNode,player_id, num_simulations = 100, c_param = 0.1) -> None:
#             self.root = root
#             self.num_simulations = num_simulations
#             self.c_param = c_param
#             self.player_id = player_id
            

#     def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
#         #print(f"make_move -> my player id: {self.player_id}")
#         root = MonteCarloTreeSearchNode(Board(game.get_board()), player_id=self.player_id, d=0, root_player=self.player_id,id=0,num_simulations=self.num_simulations, c_param=self.c_param)
#         best_idx = None
#         max_eval = minimax(root, 2, float('-inf'), float('+inf'), True, best_idx)
#         print(f'Mossa scelta con {-max_eval} possibili perdite su {self.num_simulations}') # Non sono convinto di num_sim
#         # STAMPATI LA BOARD E LA MOSSA CORRISPONDENTE

#         from_pos, move = root.children[best_idx].get_action()
#         #print('In make_move del nostro player -> Il nodo selezionato è il seguente: ', selected_node)
#         #print(f"In make_move del nostro player -> from_pos (col,row): {from_pos}, move: {move}")
#         return from_pos, move


if __name__ == '__main__':
    players = np.empty(2, dtype=Player)
    tot = 20
    # cross validation backbone to find best hyperparameters
    # this below is the best configuration found if we consider a performance/execution_time tradeoff
    #wins and matches for accuracy
    wins = 0
    matches = 0

    #play tot games
    for i in tqdm.tqdm(range(tot)):
        print()
        my_player_id = random.randint(0, 1)
        print(f"my_player_id: {my_player_id}")
        g = Game()
        #g.print()

        # player initialization -> our player is players[my_player_id]
        players[my_player_id] = MonteCarloPNSPlayer(player_id=my_player_id,duration=1, c_param=0.1, pn_param=1)
        players[1-my_player_id] = RandomPlayer()
        #players[1 - my_player_id] = MonteCarloPlayer(player_id=1-my_player_id,num_simulations=200, c_param=0.1)
        
        # play the game
        winner = g.play(players[0], players[1], my_player_id)
        print(f"Winner: Player {winner}")
        matches += 1

        #update accuracy
        if winner == my_player_id:
            wins += 1
        acc  = 100*float(wins)/float(matches)
        print("accuracy: ",acc )
        # Contro MCTS con ns = 200 e c = 0.1
        # io sono ns = 200, c = 0.1 e pn = 0.5
        # Su 100 parrtite -> 60 vinte