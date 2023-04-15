Pacman Capture the Flag using Monte Carlo Tree Search (MCTS)
This project is an implementation of Monte Carlo Tree Search (MCTS) algorithm to solve the classic Pacman Capture the Flag contest. The main program is pathfinder.py.

Installation
Clone the repository

bash
Copy code
git clone https://github.com/<username>/<repository-name>.git
Install dependencies

Copy code
pip install -r requirements.txt
Usage
To run the program, simply run the pathfinder.py script with the following command:

Copy code
python pathfinder.py
This will start the game with a default layout and the default number of iterations for the MCTS algorithm. You can specify the layout and the number of iterations by passing command line arguments:

css
Copy code
python pathfinder.py --layout <layout_file> --iterations <num_iterations>
For example, to run the program on smallCapture layout with 1000 iterations, you can run:

css
Copy code
python pathfinder.py --layout smallCapture --iterations 1000
Gameplay
The game is played between two teams: the red team and the blue team. Each team consists of two Pacman agents and two ghost agents. The objective of the game is to capture the opponent team's flag while protecting your own flag. The team with the highest score at the end of the game wins.

During each turn, the Pacman agents can move around the map to collect food pellets, capture the flag, or defend their own flag. The ghost agents can chase and capture opposing Pacman agents or defend their own flag.

MCTS Algorithm
The MCTS algorithm used in this project is a variant of the Upper Confidence Bounds for Trees (UCT) algorithm, which is a popular algorithm for solving games with imperfect information. The algorithm works by simulating a number of random games from the current game state and updating the tree with the results of these simulations. The algorithm then selects the action with the highest expected reward based on the information in the tree.

License
This project is licensed under the MIT License.
