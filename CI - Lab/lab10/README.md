# Tic Tac Toe Reinforcement Learning
This repository contains the implementation of a Tic Tac Toe game using Q-learning with Prioritized Experience Replay (PER) for reinforcement learning. The code is written in Python and utilizes the NumPy library for matrix operations.

## Overview
The Tic Tac Toe game is implemented as an environment where an agent learns to play against an opponent. The Q-learning algorithm is used to update the Q-values for each state-action pair, allowing the agent to make better decisions over time. The implementation also includes Prioritized Experience Replay to improve learning efficiency.

## Classes and Functions
- Memory Class (memory.py)
The Memory class is responsible for managing the Prioritized Experience Replay (PER) buffer.

1.  __init__(self, capacity, alpha=0.5) " : Initializes the memory with a given capacity and alpha value.
2.  add(self, experience): Adds an experience tuple to the memory buffer.
3. sample(self, batch_size): Samples a batch of experiences from the memory buffer.
4. print_board(pos) Function (tic_tac_toe.py)
This function prints the Tic Tac Toe board based on the current state.


- state_value(pos) Function (tic_tac_toe.py)
This function evaluates the state and returns a reward based on the game outcome.
```pyth
def state_value(pos: State):
    """Evaluate state: +1 first player wins"""

```
- my_player Function (main.py)
This function represents the agent's policy during the game, including Q-value updates.
```pyth
def my_player(q_table, state, available, trajectory, eps, discount_factor, lr):
   ```
   
- update_with_memory Function (main.py)
This function updates the Q-table using experiences from the memory buffer.
```pyth
def update_with_memory(memory, q_table, discount_factor, lr):
```

## Results
The model's performance is evaluated based on accuracy and its ability to win games against various opponents. The key results and metrics include:

Accuracy has been used as metric
- up to 95% if the my_Agent starts the game (on 100 matches)
- up to 70% is the oppositor starts the game (on 100 matches)
- up to 85% for randomic order (on 100 matches)

You can find other details looking at the code, it's full of comments

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Inspired by the concepts of reinforcement learning and Q-learning.
Special thanks to the contributors of the open-source libraries used in this project.
Feel free to contribute, report issues, or provide feedback!
