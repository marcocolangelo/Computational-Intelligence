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
from board import Board

## MCTS - Solver implementation
def MCTSSolver(N, board : Board):

    INFINITY = float('inf')
    if board.check_winner():
        return INFINITY
    elif not board.check_winner():
        return -INFINITY
    bestChild = select(N)
    N.visitCount += 1
    if bestChild.value != -INFINITY and bestChild.value != INFINITY:
        if bestChild.visitCount == 0:
            R = -playOut(bestChild)
            addToTree(bestChild)
        else:
            R = -MCTSSolver(bestChild)
    else:
        R = bestChild.value
    if R == INFINITY:
        N.value = -INFINITY
        return R
    elif R == -INFINITY:
        for child in getChildren(N):
            if child.value != R:
                R = -1
                break
        else:
            N.value = INFINITY
            return R
    N.computeAverage(R)
    return R
        
