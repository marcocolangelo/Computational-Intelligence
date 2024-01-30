# Game-Playing Agent with MCTS and PN-MCTS

## Overview

This repository contains the implementation of a game-playing agent using **Monte Carlo Tree Search (MCTS)** and an extension called **Proof-Number MCTS (PN-MCTS)**. The agent is designed to play a specific game, and the implementation includes various components such as the game board, MCTS algorithm, and configurations for experimentation.
As a result of our test results, we can say that the **PN-MCTS** version is a good step forward from the classic **MCTS** version

## Game Description

The game implemented for the agent is Quixo, a strategic board game played on a 5x5 square grid. Each player starts with a set number of pieces on the board, and the objective is to form a line of five of their own pieces either horizontally, vertically, or diagonally. The game progresses through a series of turns where players make moves to advance their position or hinder their opponent.

## Key Components

- **Board:** The Quixo board is a 5x5 grid, with each cell being either empty or occupied by a player's piece. The pieces are represented by cubes with different faces.

- **Moves:** Players can make moves by selecting a row or column and pushing it, sliding all the cubes in that row or column to the furthest empty space. The player can choose to push a row or column from either end, making strategic decisions to align their pieces.

- **Winning Condition:** The winning condition in Quixo is to create a line of five of the player's own pieces either horizontally, vertically, or diagonally. Achieving this configuration results in a win.

- **Turn-Based:** Quixo follows a turn-based structure, with each player taking turns to make a move. The game alternates between the players until a winning condition is met or the game ends in a draw.

Understanding the rules and objectives of Quixo is essential for interpreting the agent's performance and the results obtained through the Monte Carlo Tree Search (MCTS) and Proof-Number Monte Carlo Tree Search (PN-MCTS) algorithms.

## Components

### 1. board.py

The `board.py` file models the game board. Key features include:

- **Move Enum:** Defines possible moves (TOP, BOTTOM, LEFT, RIGHT).
- **Board Class:** Represents the game board and includes methods for legal actions, applying moves, checking winners, and printing the board.

### 2. game.py

The `game.py` file contains the `MonteCarloPNSPlayer` class, representing the player using PN-MCTS. The class interacts with the `play_game` function and integrates with the MCTSNode class. Key attributes include:

- `duration`: Time limit for MCTS.
- `c_param`: Parameter for balancing exploration and exploitation in MCTS.
- `pn_param`: Parameter for the PN-UCB function in selection.
- `MR_hybrid`: Flag to enable/disable Minimax Rollout hybridization.
- `minimax_depth`: Depth of the Minimax search.

It contains the `MonteCarloPlayer_classic` as well and it represents the original version of a MCTS player.

### 3. tree.py

In the `tree.py` file, we implemented a simple Monte Carlo tree search (MCTS) algorithm based on the Upper Confidence Bound for Trees (UCT) selection. The `MonteCarloTreeSearchNodeNoGreedy` class represents the Monte Carlo tree nodes. Key functions include:

- **Selection (here called best_child):** Implements the Upper Confidence Bound for Trees (UCT) formula to select the best child of a node.
- **Expansion:** Expands the search tree by adding a new child node to the current node.
- **Rollout:** Simulates a game from the current state to a terminal state using a given rollout policy.
- **Backpropagation:** Updates the number of visits and the results of the node, propagating these updates up to the root of the tree.

### 4. tree_pns.py

The `tree_pns.py` file extends the basic MCTS implementation to include Proof-Number Search (PN-MCTS). The `PN_MCTS_Node` class represents nodes in the tree. Key functions include:

- **Update Proof and Disproof Numbers:** Evaluates and updates the proof and disproof numbers of the node.
- **Rank Children:** Ranks all the children based on the type of node.
- **Final Child Selection:** Selects the child with the most visits as the final choice.

### 5. minimax_PNS.py (still work in progress)

The `minimax_PNS.py` file introduces an additional feature to the PN-MCTS player, transforming the origianal rollout function into a **MiniMax hybrid** version. The aim of this new rollout approach is to explore up to a fixed depth the subtree below a specific node in order to find information about moves bringing to **certain wins** or **certain looses** (to avoid of course).
Please, consider it just as a still work in progress feature, it still doesn't seem to be working properly.

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
| #3                | 0.5      | 0.5     | 0.1      | 100         | 67.0%               |
| #4                | 1        | 0.1     | 2.0      | 100         | 65.0%               |
| #5                | 0.5      | 0.5     | 0.5      | 1000        | 73.1%               |


### Best Configuration

The best configuration found for the game-playing agent is #5 in the MCTS configuration and #5 in the PN-MCTS configuration. These configurations achieved a victory percentage of 98.0% and over 99% (againt a Random player) respectively.
Hence, the PN-MCTS player is to consider our main choice to present you for this project.

## Testing and Experiments

Our testing process involved running multiple experiments with different configurations, including variations in the duration, C parameter, and PN parameter. We conducted cross-validation to ensure the reliability of our results.
In the `test1.py` and `test2.py` files you can find some tests between a classic version of the MCTS player against the Random player.
In the `main.py` file you can find some tests between the PN-MCTS player against the MCTS player.

## Contributing
Project coded in collaboration of Roberto Pulvirenti, a.k.a. ImBlurryF4c3 on GitHub
If you wish to contribute to the project, please follow these guidelines:

- Adhere to the coding standards specified in the project.
- Submit issues for bug reports or feature requests.
- Propose new features through pull requests, ensuring proper documentation.

## License

This project is licensed under the [MIT License](LICENSE).
