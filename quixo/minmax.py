from tree import MonteCarloTreeSearchNode
from board import Board

# Introduco un esempio di minimax
def minimax(node: MonteCarloTreeSearchNode, depth, alpha, beta, maximizing_player, best_idx) -> MonteCarloTreeSearchNode:
    # Base case: check if the game is over or depth limit reached
    if node.is_terminal_node():
        winner = node.state.check_winner()
        if winner == node.root_player:
            return 0
        else:
            return -1
    if depth == 0:
        # Faccio partire MCTS -> aka best_action()
        # L'evaluation è da fare sul rapporto tra numero vittorie/numero di sconfitte
        mcts_best_node = node.best_action() # mi torna il nodo migliore a questo livello
        # ma io sono interessato al numero di sconfitte di node
        # quindi è come se usassi best_action() giusto per propagare i risultati fino a node.
        return node.evaluation()
        
    
    if maximizing_player:
        max_eval = float('-inf')
        # Funzione che mi espande totalmente node
        node.expand_children()
        for i, child in enumerate(node.children):
            eval = minimax(child, depth - 1, alpha, beta, False, best_idx)
            if eval > max_eval:
                max_eval = eval
                best_idx = i
            if max_eval > beta:
                #print(f'BREAK IN MAXIMIZING ho tagliato {len(node.children)-counter}')
                break
            alpha = max(alpha, max_eval)
        return max_eval
    else:
        min_eval = float('inf')
        node.expand_children()
        for i, child in enumerate(node.children):
            eval = minimax(child, depth - 1, alpha, beta, True, best_idx)
            if eval < min_eval:
                min_eval = eval
                best_idx = i
            if min_eval < alpha:
                #print(f'BREAK IN MINIMIZING ho tagliato {len(node.children)-counter}')
                break
            beta = min(beta, min_eval)
        return min_eval
    




# funzione MCTS-MS(rootstate):
#     nodo radice = nodo con stato = rootstate
#     while non è il momento di fermarsi:
#         nodo = nodo radice
#         stato = rootstate
#         // Fase di selezione
#         while nodo non è una foglia:
#             se visita(nodo) > soglia e non è stato eseguito minimax:
#                 esegui minimax su nodo
#             nodo = seleziona figlio con il valore UCB più alto
#             stato = transizione(stato, azione del nodo)
#         // Fase di espansione
#         se non è stato eseguito minimax su nodo e visita(nodo) > soglia:
#             esegui minimax su nodo
#         se nodo non è terminale:
#             espandi nodo
#             nodo = uno dei figli del nodo
#             stato = transizione(stato, azione del nodo)
#         // Fase di simulazione
#         risultato simulazione = simulazione(stato)
#         // Fase di backpropagation
#         while nodo non è nullo:
#             visita(nodo) += 1
#             guadagno(nodo) += risultato simulazione
#             nodo = genitore del nodo
#     return azione del figlio del nodo radice con il guadagno più alto
