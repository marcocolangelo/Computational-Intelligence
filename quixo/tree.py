# from copy import deepcopy
# import math
# import random
# import numpy as np
# from game import Game, Move
# import itertools
# import math


# class TreeNode:
#     def __init__(self, game_state, player_id):
#         self.game_state = game_state
#         self.visits = 0
#         self.reward_sum = 0
#         self.children = []
#         self.player_id = player_id # Rappresenta il player che fa la mossa e crea lo stato per il nuovo figlio



# class MonteCarloTree:
#     # num_child è un hyperparameter
#     def __init__(self, root_state, num_child):
#         self.root = TreeNode(root_state)
#         self.n = num_child

#     def select_node(self, current_node):
#         # Implement the node selection using the UCB equation
#         selected_node = None
#         max_ucb = float('-inf')
#         for child in current_node.children:
#             ucb = child.reward_sum / child.visits + math.sqrt(2 * math.log(current_node.visits) / child.visits)
#             if ucb > max_ucb:
#                 max_ucb = ucb
#                 selected_node = child
#         return selected_node

#     # Considerando game_state come una board
#     def expand_node(self, parent_node):
#         # Add child nodes representing possible moves from the current state
#         possible_moves = self._get_possible_moves(parent_node.player_id,parent_node.game_state)
#         possible_moves = random.sample(possible_moves, self.n)
#         # might be better choose just one random move
#         for move in possible_moves:
#             from_pos, slide = move
#             new_board = deepcopy(parent_node.game_state)
#             new_board[from_pos] = parent_node.player_id # corrisponde al _take di game
#             axis_0, axis_1 = from_pos
#             # np.roll performs a rotation of the element of a 1D ndarray
#             if slide == Move.RIGHT:
#                 new_board[axis_0] = np.roll(new_board[axis_0], -1)
#             elif slide == Move.LEFT:
#                 new_board[axis_0] = np.roll(new_board[axis_0], 1)
#             elif slide == Move.BOTTOM:
#                 new_board[:, axis_1] = np.roll(new_board[:, axis_1], -1)
#             elif slide == Move.TOP:
#                 new_board[:, axis_1] = np.roll(new_board[:, axis_1], 1)
#             p_id = (parent_node.player_id + 1) % 2
#             new_node = TreeNode(new_board, p_id)
#             parent_node.children.append(new_node)

#     def simulate(self, node):
#         # Simulate the game from a state until a termination condition
#         current_state = node.game_state
#         while not current_state.is_terminal():
#             from_pos, slide = current_state.make_move(self)
#             current_state.play(from_pos, slide)

#         # Use the black-box function to simulate the remaining moves
#         current_player_idx = current_state.current_player()
#         winner = -1
#         while winner < 0:
#             from_pos, slide = current_state.make_move(self)
#             current_state.play(from_pos, slide)
#             winner = current_state.check_winner()
#             current_player_idx += 1
#             current_player_idx %= len(current_state.players)

#         return winner

#     def backpropagate(self, node, reward):
#         # Update the node statistics along the path from the leaf to the root
#         current_node = node
#         while current_node is not None:
#             current_node.visits += 1
#             current_node.reward_sum += reward
#             current_node = current_node.parent

#     def search(self, num_simulations):
#         # Perform the MCTS search for a number of simulations
#         for _ in range(num_simulations):
#             selected_node = self.select_node(self.root)
#             self.expand_node(selected_node)
#             leaf_node = self.select_node(selected_node)
#             reward = self.simulate(leaf_node)
#             self.backpropagate(leaf_node, reward)

#     def __acceptable_slides(from_position: tuple[int, int]):
#         """When taking a piece from {from_position} returns the possible moves (slides)"""
#         acceptable_slides = [Move.BOTTOM, Move.TOP, Move.LEFT, Move.RIGHT]
#         axis_0 = from_position[0]    # axis_0 = 0 means uppermost row
#         axis_1 = from_position[1]    # axis_1 = 0 means leftmost column

#         if axis_0 == 0:  # can't move upwards if in the top row...
#             acceptable_slides.remove(Move.TOP)
#         elif axis_0 == 4:
#             acceptable_slides.remove(Move.BOTTOM)

#         if axis_1 == 0:
#             acceptable_slides.remove(Move.LEFT)
#         elif axis_1 == 4:
#             acceptable_slides.remove(Move.RIGHT)
#         return acceptable_slides


#     # Implementazione di get_possible_moves come metodo privato
#     def _get_possible_moves(self, player_id, board):
#         from_all_possible_pos = list(itertools.product(list(range(5)), repeat=2))
#         possible_moves = []
#         for from_pos in from_all_possible_pos:
#             row, col = from_pos
#             from_border = row in (0, 4) or col in (0, 4)
#             if not from_border:
#                 continue  # the cell is not in the border
#             if board[from_pos] != player_id and board[from_pos] != -1:
#                 continue  # the cell belongs to the opponent
#             accettable_slides = self.__acceptable_slides(tuple([row,col]))
#             for slide in accettable_slides:
#                 possible_moves.append(tuple([row,col]), slide)
#         return possible_moves

# def main():
#     # Initialize the Monte Carlo tree
#     tree = MonteCarloTree(GameState())

#     # Perform 1000 simulations
#     for _ in range(1000):
#         # Select the root node
#         selected_node = tree.select_node(tree.root)

#         # Expand the root node
#         tree.expand_node(selected_node)

#         # Simulate the game from the expanded node
#         leaf_node = tree.select_node(selected_node)
#         reward = tree.simulate(leaf_node)

#         # Backpropagate the reward
#         tree.backpropagate(leaf_node, reward)


# if __name__ == "__main__":
#     main()



import numpy as np
from collections import defaultdict
import itertools
from game import Move
import random
from copy import deepcopy
from board import Board


class MonteCarloTreeSearchNode():
    def __init__(self, state: Board, player_id, parent=None, parent_action=None):
        self.state = state                   # The state of the board
        self.parent = parent                 # Parent node
        self.parent_action = parent_action   # None for the root node and for other nodes it is equal 
                                             # to the action which it’s parent carried out
        self.children = []                   # It contains the children nodes
        self._number_of_visits = 0           # Number of times current node is visited
        self._results = defaultdict(int)     
        self._results[1] = 0
        self._results[-1] = 0
        self.player_id = player_id           # The player who is going to carry out the move
        self._untried_actions = None
        self._untried_actions = self.untried_actions() # all possible moves from the current_state for player_id
        return


    # Returns the list of untried actions from a given state  
    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions
    

    # Returns the difference of wins/losses
    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses
    

    # Returns the number of times the node has been visited
    def n(self):
        return self._number_of_visits
    
    def expand(self):	
        action = self._untried_actions.pop()
        # Applico la mossa
        next_state = deepcopy(self.state)
        next_state.move(action, self.player_id)
        p_id = (p_id + 1) % 2
        child_node = MonteCarloTreeSearchNode(next_state, p_id, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node
    

    def is_terminal_node(self):
        if self.state.check_winner() != -1:
            return True
        return False
    

    # Corrisponde alla funzione di simulation nella implementazione precedente
    def rollout(self):
        current_rollout_state = self.state
        
        while current_rollout_state.check_winner() == -1:
            possible_moves = current_rollout_state.get_legal_actions()
            # seleziona randomicamente l'azione
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        # deve tornare il risultato del gioco (potrei utilizzare la stessa check_winner)
        return current_rollout_state.game_result()
    