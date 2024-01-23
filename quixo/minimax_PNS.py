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
        #COn questo approccio posso solo favorire le mosse a vittoria sicura ed evitare quelle a sconfitta sicura altrimenti non posso sapere nulla di più
        if depth == 0 or state.check_winner() != -1:
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
                #se value = infinito ho trovato una vittoria
                value = -value #così se almeno una perdita allora max_eval = infinito e quindi so che non è una mossa a vittoria sicura quella del nodo corrente
                max_eval = max(max_eval, value)
                # if max_eval == value:
                #     best_move = move
                # if max_eval > beta:
                #     break

                # update the best action
                #alpha = max(alpha,max_eval)

                #significa se ho trovato almeno una perdita e quindi non è una mossa a vittoria sicura
                if max_eval == float('inf'):
                    break

            if max_eval == float('inf'):
                best_move = random.choice(state.get_legal_actions(player_id))
            return max_eval,best_move 

        else:
            min_eval = float('inf')
            for move in state.get_legal_actions(player_id):     #MIN
                new_state = deepcopy(state)
                new_state.move(move, player_id)
                value,_ = self.minimax_search(new_state, depth-1, 1-player_id, alpha, beta)

                #se value = infinito ho trovato una vittoria del giocatore opposto
                value = -value
                
                min_eval = min(min_eval, value)
                # update the best action
                # if min_eval == value:
                #     best_move = move
                # if min_eval < alpha:
                #     break
                # beta = min(beta,min_eval)

                #significa se ho trovato almeno una perdita e quindi non è una mossa a vittoria sicura
                if min_eval == float('-inf'):
                    break
            if min_eval == float('-inf'):
                best_move = random.choice(state.get_legal_actions(player_id))
            return min_eval,best_move 
