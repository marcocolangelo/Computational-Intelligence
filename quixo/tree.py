import math
from game import Game, Move


import math


class TreeNode:
    def __init__(self, game_state):
        self.game_state = game_state
        self.visits = 0
        self.reward_sum = 0
        self.children = []


class MonteCarloTree:
    def __init__(self, root_state):
        self.root = TreeNode(root_state)

    def select_node(self, current_node):
        # Implement the node selection using the UCB equation
        selected_node = None
        max_ucb = float('-inf')
        for child in current_node.children:
            ucb = child.reward_sum / child.visits + math.sqrt(2 * math.log(current_node.visits) / child.visits)
            if ucb > max_ucb:
                max_ucb = ucb
                selected_node = child
        return selected_node

    def expand_node(self, parent_node):
        # Add child nodes representing possible moves from the current state
        possible_moves = parent_node.game_state.get_possible_moves()
        for move in possible_moves:
            from_pos, slide = move
            new_state = parent_node.game_state.get_board()
            new_state.play(from_pos, slide)
            new_node = TreeNode(new_state)
            parent_node.children.append(new_node)

    def simulate(self, node):
        # Simulate the game from a state until a termination condition
        current_state = node.game_state
        while not current_state.is_terminal():
            from_pos, slide = current_state.make_move(self)
            current_state.play(from_pos, slide)

        # Use the black-box function to simulate the remaining moves
        current_player_idx = current_state.current_player()
        winner = -1
        while winner < 0:
            from_pos, slide = current_state.make_move(self)
            current_state.play(from_pos, slide)
            winner = current_state.check_winner()
            current_player_idx += 1
            current_player_idx %= len(current_state.players)

        return winner

    def backpropagate(self, node, reward):
        # Update the node statistics along the path from the leaf to the root
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            current_node.reward_sum += reward
            current_node = current_node.parent

    def search(self, num_simulations):
        # Perform the MCTS search for a number of simulations
        for _ in range(num_simulations):
            selected_node = self.select_node(self.root)
            self.expand_node(selected_node)
            leaf_node = self.select_node(selected_node)
            reward = self.simulate(leaf_node)
            self.backpropagate(leaf_node, reward)


def main():
    # Initialize the Monte Carlo tree
    tree = MonteCarloTree(GameState())

    # Perform 1000 simulations
    for _ in range(1000):
        # Select the root node
        selected_node = tree.select_node(tree.root)

        # Expand the root node
        tree.expand_node(selected_node)

        # Simulate the game from the expanded node
        leaf_node = tree.select_node(selected_node)
        reward = tree.simulate(leaf_node)

        # Backpropagate the reward
        tree.backpropagate(leaf_node, reward)


if __name__ == "__main__":
    main()