from copy import deepcopy
from board import Board

# # Introduco un esempio di minimax

# # Creo una classe che rappresenta il minimax
class MiniMax():
    def __init__(self, root, depth, maximizing_player, root_player, ns, cp):
        self.root = root  # initial state of the game
        self.depth = depth  # depth limit for the minimax
        self.maximizing_player = maximizing_player # True if the player is the maximizing player, False otherwise
        self.player_id = root_player    # player_id of the player in this node
        self.root_player = root_player # player_id of the root node 
        self.ns = ns    # number of simulations for MCTS in each node
        self.cp = cp    # exploration/exploitation tradeoff for MCTS


    # qui node non viene usato ma è strano -> forse invece che Board dovremmo passare proprio il nodo da cui parte la backpropagation
    def minimax(self, node, depth, maximizing_player,alfa,beta):
        #print(f"depth: {depth} and node.children: {node.children}")
        if depth == 0 or node.children == []:
            #print(f"depth = {depth} or node.children = {len(node.children)}")
            return self.__evaluation(node)
            

        if maximizing_player:
            #print(f"depth: {depth} with num_children: {len(node.children)}")
            max_eval = float('-inf')
            for child in node.children:
                eval= self.minimax(child, depth - 1, False,alfa,beta)
                if eval != None and eval > max_eval:
                    max_eval = eval
            return max_eval
        else:
            #print(f"depth: {depth} and num_children: {len(node.children)}")
            min_eval = float('inf')
            for child in node.children:
                eval = self.minimax(child, depth - 1, True,alfa,beta)
                if eval  != None and eval < min_eval:
                    min_eval = eval
            return min_eval
        
    
    def __evaluation(self,node):
        results = node.get_results()
        if self.root_player == 0:
            wins = results[0]
            loses = results[1]
        else:
            wins = results[1]
            loses = results[0]
        
        return wins-loses  
