# SITOOOO: https://ai-boson.github.io/mcts/
import random
import numpy as np
from collections import defaultdict
from copy import deepcopy
from board import Board
import time
from enum import Enum
from minimax_PNS import MiniMax


######################################
#PN-MCTS
    
# My player's nodes are OR nodes
# The opponent's nodes are AND nodes


class PNSNodeTypes(Enum):
    OR_NODE = 0    # node belongs to my_player
    AND_NODE = 1   # node belongs to the opponent


# A tree can have three values: true, false, or unknown. In
# the case of a forced win, the tree is proven and its value is true.
# In the case of a forced loss or draw, the tree is disproven and its
# value is false. Otherwise, the value of the tree is unknown.
class PNSNodeValues(Enum):
    TRUE = 0       # proven node
    FALSE = 1      # disproven node
    UNKNOWN = 2    # unknown

# Implementation of the PN-MCTS algorithm
    # the algorithm is based on the paper "PN-MCTS: Monte Carlo Tree Search with Proof and Disproof Numbers"
    # by Cameron Browne, Edward Powley, Daniel Whitehouse, Simon Lucas, Peter I. Cowling, Philipp Rohlfshagen,
class PN_MCTS_Node():
    def __init__(self, state: Board, player_id, root_player, 
                 c_param, pn_param, parent : 'PN_MCTS_Node' = None, 
                 parent_action = None, d = 0, id = 0, duration=1,MR_hybrid = False, minimax_depth = 1):
        self.state : Board = state                              # The state of the board
        self.parent : PN_MCTS_Node = parent                     # Parent node
        self.parent_action = parent_action                      # None for the root node and for other nodes it is
                                                                # equal to the action which it’s parent carried out
        
        self.children: list[PN_MCTS_Node] = []      # It contains the children nodes
        self._number_of_visits = 0                              # Number of times current node is visited
        self._results = defaultdict(int)                        # A dictionary to retrieve the wins/losses
        self._results[0] = 0
        self._results[1] = 0
        self.player_id = player_id                              # The player who is going to carry out the move
        self.root_player = root_player                          # The player_id of the root node
        self._untried_actions = self.untried_actions()          # all possible moves from the current_state for player_id

        self.expanded = False  # flag to check if the node has been expanded (aka has child)
        self.pn = 0.0   # proof number, represents the minimum number of leaf nodes, which have to be proven 
                        # in order to prove the node.
        self.dpn = 0.0  # disproof number, represents the minimum number of leaf nodes, which have to be disproved 
                        # in order to disprove the node.
        self.rank = 0   # rank of the node, used to break ties in the selection phase
        self.type = PNSNodeTypes.OR_NODE if self.player_id == self.root_player else PNSNodeTypes.AND_NODE
        self.value: PNSNodeValues = None
        self.update_proof_disproof() # update the proof and disproof numbers of the node 
        self.evaluate() # evaluate the node

        # info per debugging
        self.depth = d    # depth of the node
        self.id = id      # number to identify the node (2 = 2nd son of the parent node)

        # hyperparameters 
        self.duration = duration                # duration of the tee search
        self.c_param = c_param                  # exploration/exploitation tradeoff
        self.pn_param = pn_param            # parameter to balance the proof and disproof numbers for the uct_pn_selection function
        self.MR_hybrid = MR_hybrid              # flag to enable/disable the MiniMax Rollout hybridization
        self.minimax_depth = minimax_depth                  # depth of the minimax search
        return


    # Returns the list of untried actions from a given state  
    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions(self.player_id)
        return self._untried_actions
    

    # Evaluate the node -> set TRUE(winning terminal node), FALSE(losing terminal node) or UNKNOWN(if not terminal node)
    def evaluate(self):
        if self.is_terminal_node():
            winner = self.state.check_winner()
            if winner == self.player_id:
                self.value = PNSNodeValues.TRUE
            else:
                self.value = PNSNodeValues.FALSE
        else:
            self.value = PNSNodeValues.UNKNOWN


    # Update of pn and dpn -> returns True if something changed, False otherwise (used for backpropagation)
    def update_proof_disproof(self):
        if self.expanded: # If the node has children
            if self.type == PNSNodeTypes.AND_NODE:  # opponent node
                proof = sum(child.pn for child in self.children)
                disproof = min(child.dpn for child in self.children)    #it wants to disproof my_agent behaviour so it looks for the minimum disproof number of the children nodes(my_player nodes in this case) so the action that this opponent can counterattack better
                if self.pn == proof and self.dpn == disproof:       #nothing changed
                    return False    #so return changed = False
                else:
                    self.pn = proof
                    self.dpn = disproof
                    return True
            elif self.type == PNSNodeTypes.OR_NODE: # my agent node
                disproof = sum(child.dpn for child in self.children)
                proof = min(child.pn for child in self.children)    #my_agent wants to prove its behaviour so it looks for the minimum proof number(action with minimun n of nodes to prove a win) of the children nodes(opponent nodes in this case) so the action that this agent can counterattack better
                if self.pn == proof and self.dpn == disproof: #nothing changed
                    return False #so return changed = False
                else: #something changed so update the proof and disproof numbers
                    self.pn = proof
                    self.dpn = disproof
                    return True #so return changed = True
        elif not self.expanded:
            if self.value == PNSNodeValues.FALSE:
                self.pn = float('inf')  #so impossible to prove as a winning node (it's already a losing node) hence the infinity number of nodes to prove it
                self.dpn = 0.0  #so it's a disproven node (it's a losing node)
            elif self.value == PNSNodeValues.TRUE:
                self.pn = 0.0 #so it's a proven node (it's a winning node)
                self.dpn = float('inf') #so impossible to disprove as a losing node (it's already a winning node) hence the infinity number of nodes to disprove it
            elif self.value == PNSNodeValues.UNKNOWN:
                self.pn = 1.0 #so atleast one node to prove it because it's unknown
                self.dpn = 1.0 #so atleast one node to disprove it because it's unknown
        return True


    # rank all the children based on the type of node -> used in the backpropagation phase if something changed
    # rank is used in the uct_pn_selection function 
    def rank_children(self):
        sorted_children = sorted(self.children, key=lambda child: child.rank)
        last_node = None
        for i, child in enumerate(sorted_children):
            if last_node is not None and self.type == PNSNodeTypes.OR_NODE and last_node.pn == child.pn:    #if the parent node is my_agent and the last node has the same proof number of the current node then they have the same rank
                child.rank = last_node.rank
            elif last_node is not None and self.type == PNSNodeTypes.AND_NODE and last_node.dpn == child.dpn: #if the parent node is the opponent and the last node has the same disproof number of the current node then they have the same rank
                child.rank = last_node.rank
            else:
                child.rank = i+1   #else the rank is the position of the node in the sorted list(+1 because the index starts from 0)
            last_node = child

    # Returns the difference of wins/loses
    # UNCHANGED from the normal version
    def q(self):
        if self.root_player == 0:
            wins = self._results[0]
            loses = self._results[1]
        else:
            wins = self._results[1]
            loses = self._results[0]
        return wins - loses
    

    # Returns the number of times the node has been visited
    # UNCHANGED from the normal version
    def n(self):
        return self._number_of_visits
    
    # As long as the value of the root is unknown, 
    # the most-promising node is expanded.
    def expand(self):
        if self.value == PNSNodeValues.UNKNOWN:	
            action = self._untried_actions.pop()
            # Applico la mossa
            next_state = deepcopy(self.state)
            next_state.move(action, self.player_id)
            p_id = 1 - self.player_id  
            child_node = PN_MCTS_Node(state=next_state, player_id=p_id, d=self.depth+1, 
                                    id=len(self.children)+1,root_player=self.root_player, parent=self, 
                                    parent_action=action, c_param=self.c_param, pn_param = self.pn_param,MR_hybrid=self.MR_hybrid,minimax_depth=self.minimax_depth)
            self.children.append(child_node)
            self.expanded = True
            return child_node
        else: # Significa che il nodo è proven o disproven
            self.expanded = True
            return self
    

    def is_terminal_node(self):
        if self.state.check_winner() != -1:
            return True
        return False
    

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]
    
    def rollout_policy2(self,current_rollout_state,p_id):
        # Use the minimax_search function to select the action. It want to avoid certain defeat and look for a certain win but if it can't find any of these it will select a random action(rollout_policy function)
        maximizing_player = True if p_id == self.root_player else False
        minimax = MiniMax(root=current_rollout_state, depth=self.minimax_depth, maximizing_player=maximizing_player, root_player=self.root_player)
        value,action = minimax.minimax_search(current_rollout_state, depth=self.minimax_depth, player_id=p_id,alpha=float('inf'),beta=float('-inf'))  # Use the minimax rollout function with a specified depth as a hyperparameter
        return value


    # Corresponds to the simulation function in the previous implementation.
    def rollout(self):
        current_rollout_state = deepcopy(self.state)
        p_id = self.player_id
        winner = -1
        #print("entra")
        #print(f"rollout -> self.player_id: {self.player_id}")
        while winner == -1:
            possible_moves = current_rollout_state.get_legal_actions(p_id)
            # seleziona randomicamente l'azione
            if self.MR_hybrid == False:
                #print("rollout policy classica")
                action = self.rollout_policy(possible_moves)
            else:
                # take only a tenth of the elements in possible_moves by randomly selecting every 10th element
                #mini_possible_moves = random.choices(possible_moves, k=int(len(possible_moves)/1)) 
                plausible_moves = []
                #print("rollout policy MR hybrid")
                action = None
                for move in possible_moves:
                    #print(f"mosse possibili: {possible_moves} e ora sto valutando la mossa: {move}")
                    new_state = deepcopy(current_rollout_state)
                    new_state.move(move, p_id)
                    value = self.rollout_policy2(new_state,p_id)
                    if self.minimax_depth % 2 == 1:
                        value = -value
                    if value == -float('inf'):
                       # print("trovata mossa a vittoria sicura")
                        action = move
                        break
                    if value == -1:
                        #print("trovata mossa incerta")
                        plausible_moves.append(move)
                    #if value == float('inf'):   #solo sconfitte
                        #print("trovata mossa a sconfitta sicura")

                if action == None:
                    action = self.rollout_policy(plausible_moves)        
            current_rollout_state.move(action, p_id)
            #current_rollout_state.printami()
            winner = current_rollout_state.check_winner()
            p_id = 1 - p_id
        # must return the result of the game (I could use the same check_winner)
        #print("esco")
        return winner
    

    # In this step all the statistics for the nodes are updated. Until the parent node is reached, 
    # the number of visits for each node is incremented by 1. If the result is 1, that is it 
    # resulted in a win, then the win is incremented by 1. Otherwise if result is a loss, 
    # then loss is incremented by 1.
    # It updates also the rank of the children and the proof and disproof numbers
    def backpropagate(self, result, first_node, changed):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if not first_node:
            if changed:
                changed = self.update_proof_disproof()
                if len(self.children) > 0:
                    self.rank_children()
        else:
            first_node = False
        if self.parent:
            changed = self.parent.backpropagate(result, first_node, changed)
        return changed

    
    def is_fully_expanded(self):
        return len(self._untried_actions) == 0
    

    def uct_pn_selection(self):
        tot = len(self.children)
        child_weights = [c.q()/c.n() + 
                         (self.c_param * np.sqrt((2 * np.log(self.n()) / c.n()))) +
                         self.pn_param * (1-(c.rank/tot))
                         for c in self.children]
        return self.children[np.argmax(child_weights)]
    

    # Selects node to run rollout
    # UNCHANGED (cambia solo la funzione di selection uct_pn_selection())
    def _tree_policy(self):
        current_node = self
        #print(f"Profondita': {current_node.depth} e id: {current_node.id}")
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.uct_pn_selection()
        return current_node
    

    # This is the best action function which returns the node corresponding to best possible move. 
    # The step of expansion, simulation and backpropagation are carried out by this function
    # CHANGED for PCN
    def best_action(self):
        start = time.time()
        #for _ in range(30):  
        while time.time() - start < self.duration: 
            v = self._tree_policy()
            if not v.is_terminal_node(): # If it is not a terminal node I start the rollout.
                #print(v)
                reward = v.rollout()
            else:
                reward = v.state.check_winner()
            changed = v.backpropagate(reward, True, True)
            if changed:
                for child in self.children:
                    if child.pn == 0:
                        return child
        return self.final_child_selection()
    

    # Robust child implementation.
    # you select the child with the most visits because it is the most robust one
    # more visits -> more used -> more robust
    def final_child_selection(self):
        num_visits = [c.n() for c in self.children]
        return self.children[np.argmax(num_visits)]
    

    def __str__(self):
        ascii_val = 65 # corrisponde ad A
        if self.depth == 0:
            return f'Nodo {chr(ascii_val+self.depth) + str(self.id)}'
        return f'Nodo {chr(ascii_val+self.parent.depth)+str(self.parent.id)+chr(ascii_val+self.depth) + str(self.id)}'


    # Funzione per recuperare l'azione che ha generato tale nodo 
    def get_action(self):
        return self.parent_action 