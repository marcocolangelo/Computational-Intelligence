from copy import deepcopy
from tree import MonteCarloTreeSearchNode
from board import Board

# # Introduco un esempio di minimax

# # Creo una classe che rappresenta il minimax
class MiniMax():
    def __init__(self, state, depth, maximizing_player, root_player, ns, cp):
        self.state = state  # initial state of the game
        self.depth = depth  # depth limit for the minimax
        self.maximizing_player = maximizing_player # True if the player is the maximizing player, False otherwise
        self.player_id = root_player    # player_id of the player in this node
        self.root_player = root_player # player_id of the root node 
        self.ns = ns    # number of simulations for MCTS in each node
        self.cp = cp    # exploration/exploitation tradeoff for MCTS


    def minimax(self,state : Board, depth, maximizing_player,alfa,beta):
        #print("minimaxxxxx")
        # Base case: check if the game is over or depth limit reached
        if depth == 0 or (state.check_winner() != -1):
            #print(f"in minimax ->depth: {depth} and state.check_winner(): {state.check_winner()}")
            # if max depth reached or game is over, run MCTS (if game is over, MCTS will handle it don't worry)
            root = MonteCarloTreeSearchNode(state=state, player_id=self.player_id, d=0, id=0, root_player=self.player_id, num_simulations=self.ns, c_param=self.cp)
            best_action = root.best_action()
            eval = self.__evaluation(root)
            return eval,best_action

        # Recursive case: if the game is not over and the depth limit is not reached
        # we need to expand the tree and call the minimax on the children

        # if the player is the maximizing player
        if maximizing_player:
            print(f"in minimax ->depth: {depth} and maximizing_player: {maximizing_player}")
            max_eval = float('-inf')
            max_action = None
            counter = 0
            # for each legal action in the state we need to create a new state and call the minimax on it with depth - 1 and maximizing_player = False (because we are in the minimizing player then)
            for action in state.get_legal_actions(self.player_id):
                counter += 1
                new_state = deepcopy(state)
                new_state.move(action, self.player_id) # player_id = 0 because we are in the maximizing player so the opponent
                eval,new_action = self.minimax(new_state, depth - 1, False,alfa,beta)
                max_eval = max(max_eval, eval)
                if max_eval == eval:
                    max_action = action
                if max_eval > beta:
                    break
                # update the best action
                alfa = max(alfa,max_eval)
                
            return max_eval,max_action
        else:
            print(f"in minimax ->depth: {depth} and maximizing_player: {maximizing_player}")
            # if the player is the minimizing player 
            self.player_id = 1- self.player_id
            min_eval = float('inf')
            min_action = None
            # for each legal action in the state we need to create a new state and call the minimax on it with depth - 1 and maximizing_player = True (because we are in the maximizing player then)
            for action in state.get_legal_actions(self.player_id):
                new_state = deepcopy(state)
                new_state.move(action, self.player_id) # player_id = 1 because we are in the minimize player so our player
                eval, new_action = self.minimax(new_state, depth - 1, True,alfa,beta)
                min_eval = min(min_eval, eval)
                # update the best action
                if min_eval == eval:
                    min_action = action
                if min_eval < alfa:
                    break
                beta = min(beta,min_eval)
            return min_eval,min_action
        

    def __evaluation(self,node : MonteCarloTreeSearchNode):
        results = node.get_results()
        if self.root_player == 0:
            wins = results[0]
            loses = results[1]
        else:
            wins = results[1]
            loses = results[0]
        return -loses  


# if __name__ == "__main__":
#     # Create an instance of the game
#     game = Game()

#     # Create an initial board state
#     initial_state = Board()

#     # Create an instance of the MiniMax class
#     minimax = MiniMax(initial_state, depth=5, maximizing_player=True, root_player=0, ns=100, cp=1.0)

#     # Call the minimax method to get the best action
#     best_action = minimax.minimax(initial_state, depth=3, maximizing_player=True)

#     # Print the best action
#     print("Best Action:", best_action)

#     # Print pos and move of the best action
#     from_pos, move = best_action.parent_action
#     print("From pos:", from_pos)
#     print("Move:", move)
