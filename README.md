# Pacman Capture the Flag using Monte Carlo Tree Search (MCTS)

This project is an implementation of Monte Carlo Tree Search (MCTS) algorithm to solve the classic Pacman Capture the Flag contest. The agent program is `pathfinder.py` and the game program is `capture.py`

## Installation

Clone the repository:
git clone https://github.com/RylinnM/MCTS-in-Pacman-Capture-The-Flag-environment.git


## Usage

To run the program, simply run the `capture.py` script with the following command:

`python capture.py --red=pathfinder`

This will start the game with a default layout and the default number of iterations for the MCTS algorithm. You can specify the layout and the number of iterations by passing command line arguments:


## Gameplay

The game is played between two teams: the red team and the blue team. Each team consists of two Pacman agents and two ghost agents. The objective of the game is to capture the opponent team's flag while protecting your own flag. The team with the highest score at the end of the game wins.

During each turn, the Pacman agents can move around the map to collect food pellets, capture the flag, or defend their own flag. The ghost agents can chase and capture opposing Pacman agents or defend their own flag.

## MCTS Algorithm

The MCTS algorithm used in this project is a variant of the Upper Confidence Bounds for Trees (UCT) algorithm, which is a popular algorithm for solving games with imperfect information. The algorithm works by simulating a number of random games from the current game state and updating the tree with the results of these simulations. The algorithm then selects the action with the highest expected reward based on the information in the tree.

## License

This project is licensed under the MIT License.

