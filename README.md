# Connect-4 AI
An Artificial Intelligence version of the classic Connect-4 classic game, using Python.

In my code I implemneted the following algorithms: MiniMax, AlphaBeta Pruning and the ExpectiMax.

<img width="280" alt="connect-4" src="https://user-images.githubusercontent.com/112930532/208289090-c3149e99-2ef7-4cf2-be48-bdaddfc69873.png">

Connect Four is a two player connection game on a 6x7 board. The goal of the game is to strategically insert a disk in one of the seven columns giving you a higher chance to connect 4 disks by row, column, or diagonal. Players alternate turns.

To play the game, run the connect4.py file.

Change this variables as you wish:
graphicMode - True for graphic mode, False for textual mode.
gameMode will be the value of 2 for player vs. player, or 1 for player vs. the chosen AI agent.
depth - the max depth to explore the minimax tree.
type - the name of the agent that will play as AI_agent if gameMode = 2.
The types are: "BestRandom", "MinimaxAgent", "AlphaBetaAgent", "ExpectimaxAgent" .
