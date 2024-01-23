# SITOOOO: https://ai-boson.github.io/mcts/
import numpy as np
from collections import defaultdict
from copy import deepcopy
from board import Board
import time
from enum import Enum
from minimax_PNS import MiniMax


######################################
#PN-MCTS
    
# I nodi del mio player sono nodi OR
# I nodi dell'avversario sono nodi AND


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
                 parent_action = None, d = 0, id = 0, duration=1,MR_hybrid = False):
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
        self.rank = 0
        self.type = PNSNodeTypes.OR_NODE if self.player_id == self.root_player else PNSNodeTypes.AND_NODE
        self.value: PNSNodeValues = None
        self.update_proof_disproof()
        self.evaluate()

        # info per debugging
        self.depth = d    # depth of the node
        self.id = id      # number to identify the node (2 = 2nd son of the parent node)

        # hyperparameters (aggiunti da Marco)
        self.duration = duration                # duration of the tee search
        self.c_param = c_param                  # exploration/exploitation tradeoff
        self.pn_param = pn_param
        self.MR_hybrid = MR_hybrid              # flag to enable/disable the MF hybridization
        return


    # Returns the list of untried actions from a given state  
    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions(self.player_id)
        return self._untried_actions
    

    # Evaluate the node -> set TRUE, FALSE or UNKNOWN
    def evaluate(self):
        if self.is_terminal_node():
            winner = self.state.check_winner()
            if winner == self.player_id:
                self.value = PNSNodeValues.TRUE
            else:
                self.value = PNSNodeValues.FALSE
        else:
            self.value = PNSNodeValues.UNKNOWN


    # Update of pn and dpn 
    def update_proof_disproof(self):
        if self.expanded: # If the node has children
            if self.type == PNSNodeTypes.AND_NODE:
                proof = sum(child.pn for child in self.children)
                disproof = min(child.dpn for child in self.children)
                if self.pn == proof and self.dpn == disproof:
                    return False
                else:
                    self.pn = proof
                    self.dpn = disproof
                    return True
            elif self.type == PNSNodeTypes.OR_NODE:
                disproof = sum(child.dpn for child in self.children)
                proof = min(child.pn for child in self.children)
                if self.pn == proof and self.dpn == disproof:
                    return False
                else:
                    self.pn = proof
                    self.dpn = disproof
                    return True
        elif not self.expanded:
            if self.value == PNSNodeValues.FALSE:
                self.pn = float('inf')
                self.dpn = 0.0
            elif self.value == PNSNodeValues.TRUE:
                self.pn = 0.0
                self.dpn = float('inf')
            elif self.value == PNSNodeValues.UNKNOWN:
                self.pn = 1.0
                self.dpn = 1.0
        return True


    # rank all the children based on the type of node
    def rank_children(self):
        sorted_children = sorted(self.children, key=lambda child: child.rank)
        last_node = None
        for i, child in enumerate(sorted_children):
            if last_node is not None and self.type == PNSNodeTypes.OR_NODE and last_node.pn == child.pn:
                child.rank = last_node.rank
            elif last_node is not None and self.type == PNSNodeTypes.AND_NODE and last_node.dpn == child.dpn:
                child.rank = last_node.rank
            else:
                child.rank = i+1
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
                                    parent_action=action, c_param=self.c_param, pn_param = self.pn_param,MR_hybrid=self.MR_hybrid)
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
        # Usa la funzione minimax_search per selezionare l'azione
        minimax = MiniMax(root=current_rollout_state, depth=4, maximizing_player=True, root_player=p_id)
        _,action = minimax.minimax_search(current_rollout_state, depth=4, player_id=p_id,alpha=float('inf'),beta=float('-inf'))  # Usa la funzione di rollout minimax con una profondità di 3
        return action


    # Corrisponde alla funzione di simulation nella implementazione precedente
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
                # print("rollout policy MR hybrid")
                action = self.rollout_policy2(current_rollout_state,p_id)
            current_rollout_state.move(action, p_id)
            #current_rollout_state.printami()
            winner = current_rollout_state.check_winner()
            p_id = 1 - p_id
        # deve tornare il risultato del gioco (potrei utilizzare la stessa check_winner)
        #print("esco")
        return winner
    

    # In this step all the statistics for the nodes are updated. Untill the parent node is reached, 
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
        while time.time() - start < self.duration:  
            v = self._tree_policy()
            if not v.is_terminal_node(): # Se non è un nodo terminale faccio partire il rollout
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
    

    # Implementazione Robust child
    # si seleziona il figlio con il maggior numero di visite
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