#preudo codice di MCTS
# function MCTS(rootstate, itermax, evaluation)
#     rootnode = Node(state = rootstate)

#     for i in range(itermax):
#         node = rootnode
#         state = rootstate.Clone()

#         # Select
#         while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
#             node = node.SelectChild()
#             state.DoMove(node.move)

#         # Expand
#         if node.untriedMoves != []:  # if we can expand (i.e. state/node is non-terminal)
#             m = random.choice(node.untriedMoves) 
#             state.DoMove(m)
#             node = node.AddChild(m, state)  # add child and descend tree

#         # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
#         while state.GetMoves() != []:  # while state is non-terminal
#             state.DoMove(random.choice(state.GetMoves()))

#         # Backpropagate
#         while node != None:  # backpropagate from the expanded node and work back to the root node
#             node.Update(state.GetResult(node.playerJustMoved))  # state is terminal. Update node with result from POV of node.playerJustMoved
#             node = node.parentNode

#     return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move  # return the move that was most visited

## approccio MonteCarlo Tree Search
import numpy as np
from game import Game, Player, Move
import random
from copy import deepcopy

class RandomPlayer(Player):
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        # Scegli una mossa casuale tra le mosse possibili
        possible_moves = game.get_possible_moves()
        return random.choice(possible_moves)

class MonteCarloPlayer(Player):
    def __init__(self, num_simulations):
        super().__init__()
        self.num_simulations = num_simulations

    def make_move(self, game: Game):
        # Ottieni l'elenco delle mosse possibili
        possible_moves = game.get_possible_moves()

        best_win_rate = -1
        best_move = None

        # Esegui la ricerca Monte Carlo per ogni mossa possibile
        for move in possible_moves:
            best_move = self.__simulate_move(game, move, best_win_rate, best_move)

        return best_move

    def __simulate_move(self, game, move, best_win_rate, best_move):
        win_count = 0

        # Simulate the game for the current move
        for _ in range(self.num_simulations):
            simulated_game = deepcopy(game)
            ok = simulated_game.play(move, 0)  # Assume this player is player 0
            if ok:
                winner = simulated_game.play_game(RandomPlayer(), RandomPlayer())  # Play the game with two dummy players
                if winner == 0:  # If this player wins
                    win_count += 1

        # If the win rate for this move is better than the current best move, update the best move
        win_rate = win_count / self.num_simulations
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_move = move

        return best_move
    
## approccio MonteCarlo Fist Visit
    class MonteCarloPlayer(Player):
        def __init__(self, num_simulations=1000, epsilon=0.1):
            super().__init__()
            self.num_simulations = num_simulations
            self.epsilon = epsilon
            self.value_dictionary = {}

        def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
            best_value = -float('inf')
            best_move = None

            # For each possible move on the border
            for x in range(5):
                for y in [0, 4]:  # Only consider the top and bottom rows
                    for move in Move:
                        best_move = self.__update_value(game, x, y, move, best_value, best_move)
                for y in range(5):
                    for x in [0, 4]:  # Only consider the left and right columns
                        for move in Move:
                            best_move = self.__update_value(game, x, y, move, best_value, best_move)

            return best_move

        def __update_value(self, game, x, y, move, best_value, best_move):
            # Simulate the game for the current move
            simulated_game = deepcopy(game)
            ok = simulated_game.__move((x, y), move, 0)  # Assume this player is player 0
            if ok:
                winner = simulated_game.play(MonteCarloPlayer(0), MonteCarloPlayer(0))  # Play the game with two dummy players
                if winner == 0:  # If this player wins
                    reward = 1
                else:
                    reward = -1

                # Update the value of the state
                state = (x, y, move)
                if state not in self.value_dictionary:
                    self.value_dictionary[state] = 0
                self.value_dictionary[state] = (1 - self.epsilon) * self.value_dictionary[state] + self.epsilon * reward

                # If the value of this state is better than the current best move, update the best move
                if self.value_dictionary[state] > best_value:
                    best_value = self.value_dictionary[state]
                    best_move = ((x, y), move)

            return best_move
        



## approccio su Model Based Learning Systems 
        
## Search with Amortized Value Estimates (SAVE) implementation 
# Quixo implementation of the SAVE algorithm
# https://arxiv.org/abs/1902.10565

        
