from tree import MonteCarloTreeSearchNode

# Introduco un esempio di minimax
def minimax(node: MonteCarloTreeSearchNode, depth, alpha, beta, maximizing_player) -> MonteCarloTreeSearchNode:
    # Base case: check if the game is over or depth limit reached
    if node.is_terminal_node():
        return node
    if depth == 0:
        # Faccio partire MCTS -> aka best_action()
        # L'evaluation è da fare sul rapporto tra numero vittorie/numero di sconfitte
        mcts_best_node = node.best_action() # mi torna il nodo migliore a questo livello
        # ma io sono interessato al rapporto vittorie sconfitte di node
        # quindi è come se usassi best_action() giusto per propagare i risultati fino a node.
        return node
        
    
    if maximizing_player:
        max_eval = float('-inf')
        # Funzione che mi espande totalmente node
        node.expand_children()
        best_node = None
        counter = 0
        for child in node.children:
            counter+= 1
            mcts_applied = minimax(child, depth - 1, alpha, beta, False)
            if mcts_applied.evaluation() > max_eval:
                max_eval = mcts_applied.evaluation()
                best_node = mcts_applied
            if max_eval > beta:
                #print(f'BREAK IN MAXIMIZING ho tagliato {len(node.children)-counter}')
                break
            alpha = max(alpha, max_eval)
        return best_node
    else:
        min_eval = float('inf')
        node.expand_children()
        counter = 0 
        for child in node.children:
            counter+=1
            mcts_applied = minimax(child, depth - 1, alpha, beta, True)
            if mcts_applied.evaluation() < min_eval:
                min_eval = mcts_applied.evaluation()
                best_node = mcts_applied
            if min_eval < alpha:
                #print(f'BREAK IN MINIMIZING ho tagliato {len(node.children)-counter}')
                break
            beta = min(beta, min_eval)
        return best_node