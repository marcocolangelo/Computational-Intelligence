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
        #print("depth: ", depth)
        #COn questo approccio posso solo favorire le mosse a vittoria sicura ed evitare quelle a sconfitta sicura altrimenti non posso sapere nulla di più
        if depth == 0 or state.check_winner() != -1:
            #print(f"depth: {depth} e winner: {state.check_winner()} with root : {self.root_player}")
            if (state.check_winner() == self.root_player):
                return float('inf'),None
            elif (state.check_winner() == 1 - self.root_player):
                return float('-inf'),None
            else:
                return -1,None

        best_move = None
        if player_id == self.root_player:    #MAX   
            max_eval = float('-inf')
            for move in state.get_legal_actions(player_id):
                new_state = deepcopy(state)
                new_state.move(move, player_id)
                value,_ = self.minimax_search(new_state, depth-1, 1-player_id, alpha, beta)
                #if value = infinity I found a victory
                value = -value        #means if I have found at least one loss or uncertain state and therefore it is not a sure win move, I can discard the node

                
                #means if I have found at least one loss or uncertain state and therefore it is not a sure win move, I can discard the node
                if value == -1 or value == 1:
                     return value,move
                     

                max_eval = max(max_eval, value)
                # if max_eval == value:
                #     best_move = move
                # if max_eval > beta:
                #     break

                # update the best action
                #alpha = max(alpha,max_eval)

               

            if max_eval == float('inf') or max_eval == -1:
                best_move = random.choice(state.get_legal_actions(player_id))
            return max_eval,best_move 

        else:
            min_eval = float('inf')
            for move in state.get_legal_actions(player_id):     #MIN
                new_state = deepcopy(state)
                new_state.move(move, player_id)
                value,_ = self.minimax_search(new_state, depth-1, 1-player_id, alpha, beta)

                #if value = infinite I found a win from the opposite player.
                value = -value

                if value == -1 or value == 1:
                     return value,move
                
                #means if I have found at least one loss or uncertain state and therefore it is not a sure win move
                min_eval = min(min_eval, value)
                # update the best action
                # if min_eval == value:
                #     best_move = move
                # if min_eval < alpha:
                #     break
                # beta = min(beta,min_eval)

                
                # if min_eval == -1:
                #     break
            if min_eval == float('-inf') or min_eval == -1:
                best_move = random.choice(state.get_legal_actions(player_id))
            return min_eval,best_move 
