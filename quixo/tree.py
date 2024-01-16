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


# SITOOOO: https://ai-boson.github.io/mcts/
import numpy as np
from collections import defaultdict
from copy import deepcopy
from board import Board



class MonteCarloTreeSearchNode():
    def __init__(self, state: Board, player_id, d, id, root_player,parent : 'MonteCarloTreeSearchNode' = None, parent_action=None, num_simulations=100, c_param=0.1):
        self.state : Board = state                   # The state of the board
        self.parent : MonteCarloTreeSearchNode = parent                 # Parent node
        self.parent_action = parent_action   # None for the root node and for other nodes it is equal 
                                             # to the action which it’s parent carried out
        self.children: list[MonteCarloTreeSearchNode] = []  # It contains the children nodes
        self._number_of_visits = 0           # Number of times current node is visited
        self._results = defaultdict(int)     
        self._results[0] = 0
        self._results[1] = 0
        self.player_id = player_id           # The player who is going to carry out the move
        self.root_player = root_player       # The player_id of the root node
        #print("init di node -> my player_id: ", self.player_id)
        #self._untried_actions = None
        self._untried_actions = self.untried_actions() # all possible moves from the current_state for player_id

        # info per debugging
        self.depth = d
        self.id = id      # numero del figlio

        # hyperparameters (aggiunti da Marco)
        self.num_simulations = num_simulations  # number of simulations
        self.c_param = c_param  # exploration/exploitation tradeoff
        return
    
    def get_results(self):
        return self._results[0],self._results[1]

    # Returns the list of untried actions from a given state  
    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions(self.player_id)
        #print(self._untried_actions)
        return self._untried_actions
    

    # Returns the difference of wins/losses
    def q(self):
        if self.root_player == 0:
            wins = self._results[0]
            loses = self._results[1]
        else:
            wins = self._results[1]
            loses = self._results[0]
        return wins - loses
    

    # Returns the number of times the node has been visited
    def n(self):
        return self._number_of_visits
    
    def expand(self):	
        action = self._untried_actions.pop()
        # Applico la mossa
        next_state = deepcopy(self.state)
        next_state.move(action, self.player_id)
        p_id = 1 - self.player_id  # prima era p_id = 1 - p_id e non funzionava per cui ho cambiato come vedi qui
        #print(f"expand -> p_id: {p_id} e self.player_id: {self.player_id}")
        child_node = MonteCarloTreeSearchNode(state=next_state, player_id=p_id, d=self.depth+1, id=len(self.children)+1,root_player=self.root_player, parent=self, parent_action=action, num_simulations=self.num_simulations, c_param=self.c_param)
        self.children.append(child_node)
        return child_node
    

    def is_terminal_node(self):
        if self.state.check_winner() != -1:
            return True
        return False
    

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]


    # Corrisponde alla funzione di simulation nella implementazione precedente
    def rollout(self):
        current_rollout_state = deepcopy(self.state)
        p_id = self.player_id
        #print(f"rollout -> self.player_id: {self.player_id}")
        while current_rollout_state.check_winner() == -1:
            possible_moves = current_rollout_state.get_legal_actions(p_id)
            # seleziona randomicamente l'azione
            action = self.rollout_policy(possible_moves)
            current_rollout_state.move(action, p_id)
            #current_rollout_state.printami()
            p_id = 1 - p_id
        # deve tornare il risultato del gioco (potrei utilizzare la stessa check_winner)
        return current_rollout_state.check_winner()
    

    # In this step all the statistics for the nodes are updated. Untill the parent node is reached, 
    # the number of visits for each node is incremented by 1. If the result is 1, that is it 
    # resulted in a win, then the win is incremented by 1. Otherwise if result is a loss, 
    # then loss is incremented by 1.
    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    
    def is_fully_expanded(self):
        return len(self._untried_actions) == 0
    
    # Credo sia la UCB equation (confermo che è la UCB equation (by Marco))
    def best_child(self, c_param=0.1,iter_sim=10):
        # approccio eps-greedy per la scelta del c_param
        #print(f"iter_sim: {iter_sim} e num_simulations: {self.num_simulations}")
        if iter_sim/self.num_simulations < 0.25:
            #print(f"iter_sim: {iter_sim} e num_simulations: {self.num_simulations}")
            c_param = self.c_param * 5
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) 
                           for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    # Selects node to run rollout
    def _tree_policy(self, iter_sim):
        current_node = self
        #print(f"Profondita': {current_node.depth} e id: {current_node.id}")
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child(c_param = self.c_param,iter_sim=iter_sim)
        return current_node
    
    # This is the best action function which returns the node corresponding to best possible move. 
    # The step of expansion, simulation and backpropagation are carried out by this function
    def best_action(self):
        simulation_no = self.num_simulations    # l'ho reso un iperparametro così possiamo fare prove con varie configurazioni
        for i in range(simulation_no):
            v = self._tree_policy(i)
            #print(v)
            reward = v.rollout()
            v.backpropagate(reward)
        
        # non mi convince la scelta di mettere c_param = 0 quindi l'ho reso un iperparametro e provato a metterlo a 0.1
            # valutiamo la scelta di farlo variare nel tempo per favorire più l'exploration all'inizio e più la exploitation alla fine
        return self.best_child(c_param = self.c_param,iter_sim=simulation_no)
    
    def __str__(self):
        ascii_val = 65 # corrisponde ad A
        return f'Nodo {chr(ascii_val+self.depth) + str(self.id)}'

    
    def main(self):
        root = MonteCarloTreeSearchNode(Board(),1, 0, 0,root_player=1)
        selected_node = root.best_action()
        from_pos, move = selected_node.parent_action
        #print('Il nodo selezionato è il seguente: ', selected_node)
        #print(f"from_pos: {from_pos}, move: {move}")
        return 
    
if __name__ == '__main__':  
    mc = MonteCarloTreeSearchNode(Board(),1, 0, 0,root_player=1)
    mc.main()
    