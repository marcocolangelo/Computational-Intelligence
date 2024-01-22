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


## approccio su Model Based Learning Systems 
        
## Search with Amortized Value Estimates (SAVE) implementation 
# Quixo implementation of the SAVE algorithm
# https://arxiv.org/abs/1902.10565
from collections import defaultdict
from copy import deepcopy
import numpy as np
from board import Board

## MCTS - Solver implementation
class MCTSSolverNode():
    def __init__(self, state: Board, player_id, d, id, root_player,parent : 'MCTSSolverNode' = None, parent_action=None, num_simulations=100, c_param=0.1):
        self.state : Board = state                   # The state of the board
        self.parent : MCTSSolverNode = parent                 # Parent node
        self.parent_action = parent_action   # None for the root node and for other nodes it is equal 
                                             # to the action which it’s parent carried out
        self.children: list[MCTSSolverNode] = []  # It contains the children nodes
        self._number_of_visits = 0           # Number of times current node is visited
        self._results = defaultdict(int)     
        self._results[0] = 0
        self._results[1] = 0
        self.value = 0
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
        child_node = MCTSSolverNode(state=next_state, player_id=p_id, d=self.depth+1, id=len(self.children)+1,root_player=self.root_player, parent=self, parent_action=action, num_simulations=self.num_simulations, c_param=self.c_param)
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
        #if iter_sim/self.num_simulations < 0.25:
            #print(f"iter_sim: {iter_sim} e num_simulations: {self.num_simulations}")
            #c_param = self.c_param * 5
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) 
                           for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    # Selects node to run rollout
    def _tree_policy(self, iter_sim) -> 'MCTSSolverNode':
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
    
    def getChildren(self):
        return self.children

  
    def computeAverage(self, R):
      # è la funzione computeAverage(self,R) della variante Solver di MCTS  
        self.value = (self.value + R) / self._number_of_visits
        

    

class MCTSSolverTree():
    def __init__(self, root : MCTSSolverNode, player_id) -> None:
        self.root = root
        self.tree = [root]
        self.root_player_id = player_id

    def MCTSSolver(self,N : MCTSSolverNode, board : Board):

        INFINITY = float('inf')
        if board.check_winner() == self.root_player_id:
            return INFINITY
        elif board.check_winner() == (1 - self.root_player_id):
            return -INFINITY
        bestChild : MCTSSolverNode = N._tree_policy()
        N._number_of_visits += 1
        if bestChild.value != -INFINITY and bestChild.value != INFINITY:
            if bestChild._number_of_visits == 0:
                R = -bestChild.rollout()
            else:
                R = -self.MCTSSolver(bestChild)
        else:
            R = bestChild.value
        if R == INFINITY:
            N.value = -INFINITY
            return R
        elif R == -INFINITY:
            for child in N.getChildren():
                if child.value != R:
                    R = -1
                    break
                
            N.value = INFINITY
            return R
            
        N.computeAverage(R)
        return R
    

    

    

            
