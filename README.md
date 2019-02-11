# Tic-Tac-Toe Reinforcement Learning with Q-Learning and SARSA

## Goal
Using reinforcement learning (Q-learning and SARSA) to see if an AI agent can learn playing a fork (two non-blocked lines of two) in tic-tac-toe.

Example of forks:

![alt text](./readme_images/tictactoe_fork.png "Fork")



Note this program is not a game a user can play interacting with UI.

---
## How this works
An AI agent plays tic-tac-toe against an algorithm player many times. In order to win the AI agent always opens a game. The algorithm player is not an AI. It has only two simple strategies; win if it has two marks inline; block if the AI agent has two marks inline. It plays randomly for the rest. The only way for the AI agent to win a game is to play a fork.

Each execution of the program runs a certain number of episodes (games). After each episode the AI agent will receive a reward value; 1 if it wins; 0 if the game is tied; -1 if it loses. At the end of the execution, it will display a graph that shows relationship between episodes and rewards. For better visualization, the graph is smoothed over a window of size 50.

![alt text](./readme_images/tictactoe_graph.png "Graph")

---
## How to run
This is a python 3 program. Execute tictactoe_main.py.

To switch between Q-Learning and SARSA, in the main function in tictactoe_main.py, you will find these two sets of two lines.
```python
    agent = TicTacToeQLearningAgent()
    # agent = TicTacToeSarsaAgent()

    experiment = QLearningExperiment(env, agent, before_episode_callback, after_episode_callback)
    # experiment = SarsaExperiment(env, agent, before_episode_callback, after_episode_callback)
```
Comment out whichever you don't want.

---
## Folder hierarchy
* **the root folder**: contains python script files with the tic-tac-toe specific code.
* **readme_images**: contains images for this README file.
* **rl_base**: contains python script files with reinforcement learning code. These files do not have any tic-tac-toe related code. They can be reused for other reinforcement learning programs.
* **tests**: contains unit test files with pytest.
