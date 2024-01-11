from copy import deepcopy
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
        possible_moves = parent_node.get_possible_moves(self.game_state)

        # might be better choose just one random move
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

    def __move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        '''Perform a move'''
        if player_id > 2:
            return False
        # Oh God, Numpy arrays
        prev_value = deepcopy(self._board[(from_pos[1], from_pos[0])])
        acceptable = self.__take((from_pos[1], from_pos[0]), player_id)
        if acceptable:
            acceptable = self.__slide((from_pos[1], from_pos[0]), slide)
            if not acceptable:
                self._board[(from_pos[1], from_pos[0])] = deepcopy(prev_value)
        return acceptable

    def __take(self, from_pos: tuple[int, int], player_id: int) -> bool:
        '''Take piece'''
        # acceptable only if in border
        acceptable: bool = (
            # check if it is in the first row
            (from_pos[0] == 0 and from_pos[1] < 5)
            # check if it is in the last row
            or (from_pos[0] == 4 and from_pos[1] < 5)
            # check if it is in the first column
            or (from_pos[1] == 0 and from_pos[0] < 5)
            # check if it is in the last column
            or (from_pos[1] == 4 and from_pos[0] < 5)
            # and check if the piece can be moved by the current player
        ) and (self._board[from_pos] < 0 or self._board[from_pos] == player_id)
        if acceptable:
            self._board[from_pos] = player_id
        return acceptable

    def __slide(self, from_pos: tuple[int, int], slide: Move) -> bool:
        '''Slide the other pieces'''
        # define the corners
        SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
        # if the piece position is not in a corner
        if from_pos not in SIDES:
            # if it is at the TOP, it can be moved down, left or right
            acceptable_top: bool = from_pos[0] == 0 and (
                slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is at the BOTTOM, it can be moved up, left or right
            acceptable_bottom: bool = from_pos[0] == 4 and (
                slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is on the LEFT, it can be moved up, down or right
            acceptable_left: bool = from_pos[1] == 0 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
            )
            # if it is on the RIGHT, it can be moved up, down or left
            acceptable_right: bool = from_pos[1] == 4 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
            )
        # if the piece position is in a corner
        else:
            # if it is in the upper left corner, it can be moved to the right and down
            acceptable_top: bool = from_pos == (0, 0) and (
                slide == Move.BOTTOM or slide == Move.RIGHT)
            # if it is in the lower left corner, it can be moved to the right and up
            acceptable_left: bool = from_pos == (4, 0) and (
                slide == Move.TOP or slide == Move.RIGHT)
            # if it is in the upper right corner, it can be moved to the left and down
            acceptable_right: bool = from_pos == (0, 4) and (
                slide == Move.BOTTOM or slide == Move.LEFT)
            # if it is in the lower right corner, it can be moved to the left and up
            acceptable_bottom: bool = from_pos == (4, 4) and (
                slide == Move.TOP or slide == Move.LEFT)
        # check if the move is acceptable
        acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
        # if it is
        if acceptable:
            # take the piece
            piece = self._board[from_pos]
            # if the player wants to slide it to the left
            if slide == Move.LEFT:
                # for each column starting from the column of the piece and moving to the left
                for i in range(from_pos[1], 0, -1):
                    # copy the value contained in the same row and the previous column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i - 1)]
                # move the piece to the left
                self._board[(from_pos[0], 0)] = piece
            # if the player wants to slide it to the right
            elif slide == Move.RIGHT:
                # for each column starting from the column of the piece and moving to the right
                for i in range(from_pos[1], self._board.shape[1] - 1, 1):
                    # copy the value contained in the same row and the following column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i + 1)]
                # move the piece to the right
                self._board[(from_pos[0], self._board.shape[1] - 1)] = piece
            # if the player wants to slide it upward
            elif slide == Move.TOP:
                # for each row starting from the row of the piece and going upward
                for i in range(from_pos[0], 0, -1):
                    # copy the value contained in the same column and the previous row
                    self._board[(i, from_pos[1])] = self._board[(
                        i - 1, from_pos[1])]
                # move the piece up
                self._board[(0, from_pos[1])] = piece
            # if the player wants to slide it downward
            elif slide == Move.BOTTOM:
                # for each row starting from the row of the piece and going downward
                for i in range(from_pos[0], self._board.shape[0] - 1, 1):
                    # copy the value contained in the same column and the following row
                    self._board[(i, from_pos[1])] = self._board[(
                        i + 1, from_pos[1])]
                # move the piece down
                self._board[(self._board.shape[0] - 1, from_pos[1])] = piece
        return acceptable



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