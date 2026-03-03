Youtube：https://youtu.be/501kvOIKcMg

🐰 Bunny Carrot Run – Q-Learning Maze Game

This project is a grid-based maze game controlled by a tabular Q-learning agent.
The goal of the bunny is to collect the carrot (key) first and then reach the exit safely while avoiding traps and managing HP.

🎮 Game Description

The bunny starts at the top-left corner of the maze.
⚪A carrot and an exit are placed randomly in reachable positions.
⚪The bunny must collect the carrot before going to the exit.
⚪Stepping on traps reduces HP.
⚪Picking up a medkit restores HP.
⚪The episode ends when:
    🗡The bunny reaches the exit with the carrot (success), or
    🗡HP drops to zero (failure).
Each time the game resets, a new maze layout is generated.
