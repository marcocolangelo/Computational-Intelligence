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

## approccio MonteCarlo Tree Search
import numpy as np
from game import Game, Player, Move
import random
from copy import deepcopy

class RandomPlayer(Player):
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        # Scegli una mossa casuale tra le mosse possibili
        possible_moves = game.get_possible_moves()
        return random.choice(possible_moves)

class MonteCarloPlayer(Player):
    def __init__(self, num_simulations):
        super().__init__()
        self.num_simulations = num_simulations

    def make_move(self, game: Game):
        # Ottieni l'elenco delle mosse possibili
        possible_moves = game.get_possible_moves()

        best_win_rate = -1
        best_move = None

        # Esegui la ricerca Monte Carlo per ogni mossa possibile
        for move in possible_moves:
            best_move = self.__simulate_move(game, move, best_win_rate, best_move)

        return best_move

    def __simulate_move(self, game, move, best_win_rate, best_move):
        win_count = 0

        # Simulate the game for the current move
        for _ in range(self.num_simulations):
            simulated_game = deepcopy(game)
            ok = simulated_game.play(move, 0)  # Assume this player is player 0
            if ok:
                winner = simulated_game.play_game(RandomPlayer(), RandomPlayer())  # Play the game with two dummy players
                if winner == 0:  # If this player wins
                    win_count += 1

        # If the win rate for this move is better than the current best move, update the best move
        win_rate = win_count / self.num_simulations
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_move = move

        return best_move
    
## approccio MonteCarlo Fist Visit
    class MonteCarloPlayer(Player):
        def __init__(self, num_simulations=1000, epsilon=0.1):
            super().__init__()
            self.num_simulations = num_simulations
            self.epsilon = epsilon
            self.value_dictionary = {}

        def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
            best_value = -float('inf')
            best_move = None

            # For each possible move on the border
            for x in range(5):
                for y in [0, 4]:  # Only consider the top and bottom rows
                    for move in Move:
                        best_move = self.__update_value(game, x, y, move, best_value, best_move)
                for y in range(5):
                    for x in [0, 4]:  # Only consider the left and right columns
                        for move in Move:
                            best_move = self.__update_value(game, x, y, move, best_value, best_move)

            return best_move

        def __update_value(self, game, x, y, move, best_value, best_move):
            # Simulate the game for the current move
            simulated_game = deepcopy(game)
            ok = simulated_game.__move((x, y), move, 0)  # Assume this player is player 0
            if ok:
                winner = simulated_game.play(MonteCarloPlayer(0), MonteCarloPlayer(0))  # Play the game with two dummy players
                if winner == 0:  # If this player wins
                    reward = 1
                else:
                    reward = -1

                # Update the value of the state
                state = (x, y, move)
                if state not in self.value_dictionary:
                    self.value_dictionary[state] = 0
                self.value_dictionary[state] = (1 - self.epsilon) * self.value_dictionary[state] + self.epsilon * reward

                # If the value of this state is better than the current best move, update the best move
                if self.value_dictionary[state] > best_value:
                    best_value = self.value_dictionary[state]
                    best_move = ((x, y), move)

            return best_move
        



## approccio su Model Based Learning Systems 
        
## Search with Amortized Value Estimates (SAVE) implementation 
# Quixo implementation of the SAVE algorithm
# https://arxiv.org/abs/1902.10565
        


#################################################################################################################
# #####
# MCTS Solver
# PER DEBUG
counter_inf = 0 
counter_minusInf = 0 
profondità = 0
#

INF = float('inf')

class MCTSSolverNode():
    def __init__(self, state: Board, player_id, root_player,parent : 'MCTSSolverNode' = None, parent_action=None, num_simulations=100, c_param=0.1, d=0, id=0):
        self.state : Board = state                              # The state of the board
        self.parent : MCTSSolverNode = parent         # Parent node
        self.parent_action = parent_action                      # None for the root node and for other nodes it is
                                                                # equal to the action which it’s parent carried out
        
        self.children: list[MCTSSolverNode] = []      # It contains the children nodes
        self._number_of_visits = 0                              # Number of times current node is visited
        self.player_id = player_id                              # The player who is going to carry out the move
        self.root_player = root_player                          # The player_id of the root node
        self._untried_actions = self.untried_actions()          # all possible moves from the current_state for player_id

        # info per debugging
        self.depth = d    # depth of the node
        self.id = id      # number to identify the node (2 = 2nd son of the parent node)
        # hyperparameters (aggiunti da Marco)
        self.num_simulations = num_simulations  # number of simulations
        self.c_param = c_param                  # exploration/exploitation tradeoff

        # Cambiati rispetto a Tree search Node
        self._results = 0                                       # Sommatoria dei risultati fino ad ora ottenuti
        self._value = 0
        return


    # Returns the list of untried actions from a given state  
    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions(self.player_id)
        #print(self._untried_actions)
        return self._untried_actions
    

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
        player_starting = self.player_id
        p_id = self.player_id
        winner = -1
        #print(f"rollout -> self.player_id: {self.player_id}")
        while winner == -1:
            possible_moves = current_rollout_state.get_legal_actions(p_id)
            # seleziona randomicamente l'azione
            action = self.rollout_policy(possible_moves)
            current_rollout_state.move(action, p_id)
            #current_rollout_state.printami()
            winner = current_rollout_state.check_winner()
            p_id = 1 - p_id
        # Se il vincitore è il player che ha fatto la prima mossa -> Vittoria
        # altrimenti -> sconfitta
        # tutto viene visto dalla prospettiva del player giocante
        if winner == player_starting:
            reward = 1
        else:
            reward = -1
        return reward
    

    # In this step all the statistics for the nodes are updated. Untill the parent node is reached, 
    # the number of visits for each node is incremented by 1. If the result is 1, that is it 
    # resulted in a win, then the win is incremented by 1. Otherwise if result is a loss, 
    # then loss is incremented by 1.
    def backpropagate(self, result):
        self._number_of_visits += 1.
        global counter_inf
        global counter_minusInf
        if result == INF:
            self._value = -INF
            counter_inf += 1
        elif result == -INF:
            # To prove that this node is a loss all the children have to be proven a loss
            for child in self.children:
                if child._value != result:
                    result = -1 # lost game
                    break
            counter_minusInf += 1
            self._value = INF
        else:
            self._results += result
            self._value = self._compute_avg()
        if self.parent:
            self.parent.backpropagate(-result)

    
    def is_fully_expanded(self):
        return len(self._untried_actions) == 0
    

    # Credo sia la UCB equation (confermo che è la UCB equation (by Marco))
    def best_child(self):
        choices_weights = [c._value + self.c_param * np.sqrt((2 * np.log(self.n()) / c.n())) 
                           for c in self.children]
        # n_sim = [c.n() for c in self.children]
        # max_visits = n_sim[np.argmax(n_sim)]
        # print(f'Nodo con più visite {max_visits}')
        return self.children[np.argmax(choices_weights)]
    

    # Selects node to run rollout
    def _tree_policy(self):
        current_node = self
        treshold = 20
        global profondità
        #print(f"Profondita': {current_node.depth} e id: {current_node.id}")
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded() and current_node.n() < treshold:
                return current_node.expand()
            else:
                current_node = current_node.best_child()
                if current_node.depth > profondità:
                    profondità = current_node.depth
        return current_node
    
    # This is the best action function which returns the node corresponding to best possible move. 
    # The step of expansion, simulation and backpropagation are carried out by this function
    def best_action(self):
        simulation_no = self.num_simulations 
        reward = 0 
        duration = 1
        counter = 0
        global profondità
        profondità = 0
        start_time = time.time()
        while time.time() - start_time < duration:
            counter += 1
            v = self._tree_policy()
            # Verifico se ci sono mosse perdenti/vincenti a partire da v
            if v.check_possible_win():
                reward = INF
            elif v.check_possible_lose():
                reward = -INF
            if reward != INF and reward != -INF:
                reward = v.rollout() 
                # sistemata per farmi tornare 1 in caso di vittoria del player v.player_id
                # sconfitta (-1) altrimenti
            # prima propagazione con valore di reward e non -reward perché la prima chiamata è sul nodo v, 
            # non sul suo predecessore (v.parent)
            v.backpropagate(reward)
        
        # non mi convince la scelta di mettere c_param = 0 quindi l'ho reso un iperparametro e provato a metterlo a 0.1
        # valutiamo la scelta di farlo variare nel tempo per favorire più l'exploration all'inizio e più la exploitation alla fine
        #print(f'Totale di volte che ho backpropagato INF {counter_inf}')
        #print(f'Totale di volte che ho backpropagato -INF {counter_minusInf}')
        print(f'In totale {counter} simulazioni con profondità massima di {profondità}')
        return self.secure_child()
    
    def secure_child(self):
        final_weights = [c._value + 1/c.n() for c in self.children]
        return self.children[np.argmax(final_weights)]
    

    # Funzione per recuperare l'azione che ha generato tale nodo
    def get_action(self):
        return self.parent_action


    def __str__(self):
        ascii_val = 65 # corrisponde ad A
        if self.depth == 0:
            return f'Nodo {chr(ascii_val+self.depth) + str(self.id)}'
        return f'Nodo {chr(ascii_val+self.parent.depth)+str(self.parent.id)+chr(ascii_val+self.depth) + str(self.id)}'
    

    # Funzione per calcolare il valore di value
    def _compute_avg(self):
        return self._results/self.n()
    
    # Funzione che verifica se con una singola mossa self.player_id vince
    def check_possible_win(self) -> bool:
        return self.state.playerToMoveWins(self.player_id)
    
    # Funzione che verifica se con una singola mossa self.player_id vince
    def check_possible_lose(self) -> bool:
        return self.state.playerToMoveLose(self.player_id) 

        
