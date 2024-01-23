from copy import deepcopy
import random
from board import Board


class MiniMax():
    
    def __init__(self, root, depth, maximizing_player, root_player):
        self.root = root  # initial state of the game
        self.depth = depth  # depth limit for the minimax
        self.maximizing_player = maximizing_player # True if the player is the maximizing player, False otherwise
        self.player_id = root_player    # player_id of the player in this node
        self.root_player = root_player # player_id of the root node 


    def minimax_search(self, state  : Board, depth, player_id, alpha, beta):
        if depth == 0 or state.check_winner() != -1:
            if (state.check_winner() == self.root_player):
                return float('inf'),None
            elif (state.check_winner() == 1 - self.root_player):
                return float('-inf'),None
            else:
                return 0,None

        best_move = None
        if player_id == self.root_player:
            max_eval = float('-inf')
            for move in state.get_legal_actions(player_id):
                new_state = deepcopy(state)
                new_state.move(move, player_id)
                value,_ = self.minimax_search(new_state, depth-1, 1-player_id, alpha, beta)
                # value = -value
                max_eval = max(max_eval, value)
                if max_eval == value:
                    best_move = move
                if max_eval > beta:
                    break
                # update the best action
                alpha = max(alpha,max_eval)
            if best_move is None:
                best_move = random.choice(state.get_legal_actions(player_id))
            return max_eval,best_move 

        else:
            min_eval = float('inf')
            for move in state.get_legal_actions(player_id):
                new_state = deepcopy(state)
                new_state.move(move, player_id)
                value,_ = self.minimax_search(new_state, depth-1, 1-player_id, alpha, beta)
                # value = -value
                min_eval = min(min_eval, value)
                # update the best action
                if min_eval == value:
                    best_move = move
                if min_eval < alpha:
                    break
                beta = min(beta,min_eval)
            if best_move is None:
                best_move = random.choice(state.get_legal_actions(player_id))
            return min_eval,best_move 
