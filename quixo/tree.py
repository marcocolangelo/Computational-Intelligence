# SITOOOO: https://ai-boson.github.io/mcts/
import numpy as np
from collections import defaultdict
from copy import deepcopy
from board import Board



class MonteCarloTreeSearchNode():
    def __init__(self, state: Board, player_id, d, id, root_player,parent : 'MonteCarloTreeSearchNode' = None, parent_action=None, num_simulations=100, c_param=0.1):
        self.state : Board = state                              # The state of the board
        self.parent : MonteCarloTreeSearchNode = parent         # Parent node
        self.parent_action = parent_action                      # None for the root node and for other nodes it is
                                                                # equal to the action which it’s parent carried out
        
        self.children: list[MonteCarloTreeSearchNode] = []      # It contains the children nodes
        self._number_of_visits = 0                              # Number of times current node is visited
        self._results = defaultdict(int)                        # A dictionary to retrieve the wins/losses
        self._results[0] = 0
        self._results[1] = 0
        self.player_id = player_id                              # The player who is going to carry out the move
        self.root_player = root_player                          # The player_id of the root node
        self._untried_actions = self.untried_actions()          # all possible moves from the current_state for player_id

        # info per debugging
        self.depth = d    # depth of the node
        self.id = id      # number to identify the node (2 = 2nd son of the parent node)

        # hyperparameters (aggiunti da Marco)
        self.num_simulations = num_simulations  # number of simulations
        self.c_param = c_param                  # exploration/exploitation tradeoff
        return


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
        if self.state.check_winner(self.player_id) != -1:
            return True
        return False
    

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]


    # Corrisponde alla funzione di simulation nella implementazione precedente
    def rollout(self):
        current_rollout_state = deepcopy(self.state)
        p_id = self.player_id
        winner = -1
        #print(f"rollout -> self.player_id: {self.player_id}")
        while winner == -1:
            possible_moves = current_rollout_state.get_legal_actions(p_id)
            # seleziona randomicamente l'azione
            action = self.rollout_policy(possible_moves)
            current_rollout_state.move(action, p_id)
            #current_rollout_state.printami()
            winner = current_rollout_state.check_winner(p_id)
            p_id = 1 - p_id
        # deve tornare il risultato del gioco (potrei utilizzare la stessa check_winner)
        return winner
    

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
    def best_child(self, c_param=0.1):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) 
                           for c in self.children]
        #print(f'Il Nodo {self} ha chiamato best child.\nLen di choices_weights: {len(choices_weights)}\n# children: {len(self.children)}')
        return self.children[np.argmax(choices_weights)]
    
    # Selects node to run rollout
    def _tree_policy(self):
        current_node = self
        #print(f"Profondita': {current_node.depth} e id: {current_node.id}")
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
    
    # This is the best action function which returns the node corresponding to best possible move. 
    # The step of expansion, simulation and backpropagation are carried out by this function
    def best_action(self):
        simulation_no = self.num_simulations    # l'ho reso un iperparametro così possiamo fare prove con varie configurazioni
        for _ in range(simulation_no):
            v = self._tree_policy()
            #print(v)
            reward = v.rollout()
            v.backpropagate(reward)
        
        # non mi convince la scelta di mettere c_param = 0 quindi l'ho reso un iperparametro e provato a metterlo a 0.1
        # valutiamo la scelta di farlo variare nel tempo per favorire più l'exploration all'inizio e più la exploitation alla fine
        return self.best_child(c_param = self.c_param)
    
    def __str__(self):
        ascii_val = 65 # corrisponde ad A
        if self.depth == 0:
            return f'Nodo {chr(ascii_val+self.depth) + str(self.id)}'
        return f'Nodo {chr(ascii_val+self.parent.depth)+str(self.parent.id)+chr(ascii_val+self.depth) + str(self.id)}'


    # Funzione utilizzata dal minimax per avere tutti i children già espansi e poter chiamare ricorsivamente minimax
    def expand_children(self):
        current_node = self
        if not current_node.is_terminal_node():
            while not current_node.is_fully_expanded():
                current_node.expand()


    # Funzione sfruttata nel minimax per tornare rapporto vittorie/sconfitte 
    def evaluation(self):
        if self.root_player == 0:
            wins = self._results[0]
            loses = self._results[1]
        else:
            wins = self._results[1]
            loses = self._results[0]
        return -loses



    # def main(self):
    #     root = MonteCarloTreeSearchNode(Board(),1, 0, 0,root_player=1)
    #     selected_node = root.best_action()
    #     from_pos, move = selected_node.parent_action
    #     #print('Il nodo selezionato è il seguente: ', selected_node)
    #     #print(f"from_pos: {from_pos}, move: {move}")
    #     return
    