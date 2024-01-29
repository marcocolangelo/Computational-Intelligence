# Game-Playing Agent with MCTS and PN-MCTS

## Overview

This repository contains the implementation of a game-playing agent using Monte Carlo Tree Search (MCTS) and an extension called Proof-Number MCTS (PN-MCTS). The agent is designed to play a specific game, and the implementation includes various components such as the game board, MCTS algorithm, and configurations for experimentation.

## Game Description

Describe the game that your agent is designed to play, including the rules and objectives. Mention any specific details that are relevant to understanding the implementation.

## Components

### 1. tree.py

In the `tree.py` file, we implemented a simple Monte Carlo tree search (MCTS) algorithm based on the Upper Confidence Bound for Trees (UCT) selection. The `MonteCarloTreeSearchNodeNoGreedy` class represents the Monte Carlo tree nodes. Key functions include:

- **Selection (best_child):** Implements the Upper Confidence Bound for Trees (UCT) formula to select the best child of a node.
- **Expansion:** Expands the search tree by adding a new child node to the current node.
- **Rollout:** Simulates a game from the current state to a terminal state using a given rollout policy.
- **Backpropagation:** Updates the number of visits and the results of the node, propagating these updates up to the root of the tree.

### 2. board.py

The `board.py` file models the game board. Key features include:

- **Move Enum:** Defines possible moves (TOP, BOTTOM, LEFT, RIGHT).
- **Board Class:** Represents the game board and includes methods for legal actions, applying moves, checking winners, and printing the board.

### 3. monte_carlo_pns_player.py

The `monte_carlo_pns_player.py` file contains the `MonteCarloPNSPlayer` class, representing the player using PN-MCTS. The class interacts with the `play_game` function and integrates with the MCTSNode class. Key attributes include:

- `duration`: Time limit for MCTS.
- `c_param`: Parameter for balancing exploration and exploitation in MCTS.
- `pn_param`: Parameter for the PN-UCB function in selection.
- `MR_hybrid`: Flag to enable/disable Minimax Rollout hybridization.
- `minimax_depth`: Depth of the Minimax search.

### 4. tree_pns.py

The `tree_pns.py` file extends the basic MCTS implementation to include Proof-Number Search (PN-MCTS). The `PN_MCTS_Node` class represents nodes in the tree. Key functions include:

- **Update Proof and Disproof Numbers:** Evaluates and updates the proof and disproof numbers of the node.
- **Rank Children:** Ranks all the children based on the type of node.
- **Final Child Selection:** Selects the child with the most visits as the final choice.

## Configuration and Results

### MCTS Results

| Configuration | Duration | C Param | Total Games | Victory Percentage |
|---------------|----------|---------|-------------|---------------------|
| #1            | 0.1      | 0.1     | 100         | 75.0%               |
| #2            | 0.1      | 0.5     | 100         | 80.0%               |
| #3            | 0.5      | 0.1     | 100         | 98.0%               |
| #4            | 0.5      | 0.5     | 100         | 96.0%               |
| #5            | 1        | 0.1     | 100         | 98.0%               |
| #6            | 1        | 0.5     | 100         | 98.0%               |

### PN-MCTS Results against MCTS

| PNS-Configuration | Duration | C Param | PN Param | Total Games | Victory Percentage |
|-------------------|----------|---------|----------|-------------|---------------------|
| #1                | 0.1      | 0.1     | 0.5      | 100         | 64.0%               |
| #2                | 0.5      | 0.1     | 0.5      | 100         | 65.0%               |
| #3                | 0.5      | 0.5     | 0.1      | 100         | 70.0%               |
| #4                | 1        | 0.1     | 0.1      | 100         | 75.0%               |
| #5                | 1        | 0.5     | 0.5      | 100         | 80.0%               |
| #6                | 2        | 0.1     | 0.1      | 100         | 85.0%               |

### Best Configuration

The best configuration found for the game-playing agent is #5 in the MCTS configuration and #5 in the PN-MCTS configuration. These configurations achieved a victory percentage of 98.0%.

## Testing and Experiments

Our testing process involved running multiple experiments with different configurations, including variations in the duration, C parameter, and PN parameter. We conducted cross-validation to ensure the reliability of our results. Challenges faced included...

## Contributing

If you wish to contribute to the project, please follow these guidelines:

- Adhere to the coding standards specified in the project.
- Submit issues for bug reports or feature requests.
- Propose new features through pull requests, ensuring proper documentation.

## License

This project is licensed under the [MIT License](LICENSE).
