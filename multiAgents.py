# Ariel Ashkenazy
# ID: 208465096

"""
Introduction to Artificial Intelligence, 89570, Bar Ilan University, ISRAEL

Student name:
Student ID:

"""

# multiAgents.py
# --------------
# Attribution Information: part of the code were created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# http://ai.berkeley.edu.
# We thank them for that! :)


import random
import util
import math

from connect4 import Agent, GameState


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth='2'):
        self.index = 1  # agent is always index 1
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class BestRandom(MultiAgentSearchAgent):

    def getAction(self, gameState):

        return gameState.pick_best_move()


class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.isWin():
        Returns whether or not the game state is a winning state for the current turn player

        gameState.isLose():
        Returns whether or not the game state is a losing state for the current turn player

        gameState.is_terminal()
        Return whether or not that state is terminal
        """

        def rb_minimax(state, depth):
            cur_action = None
            # reached the leafs level or 'depth' limits us from going deeper, return the heuristic evaluation
            if state.is_terminal() or depth == 0:
                return self.evaluationFunction(state), cur_action

            turn = state.turn
            # creates a list of tuples- (action, child)
            # where action is a possible action from current state
            # and child is the new state after choosing this action
            children = [(action, state.generateSuccessor(self.index, action)) for action in state.getLegalActions(turn)]
            if turn == 1:
                cur_max = float('-inf')
                for action, child in children:
                    child.switch_turn(child.turn)
                    value, _ = rb_minimax(child, depth - 1)
                    if cur_max < value:
                        cur_max = value
                        cur_action = action
                return cur_max, cur_action
            else:  # turn == 0
                cur_min = float('inf')
                for action, child in children:
                    child.switch_turn(child.turn)
                    value, _ = rb_minimax(child, depth - 1)
                    if cur_min > value:
                        cur_min = value
                        cur_action = action
                return cur_min, cur_action

        # return best_action
        _, action = rb_minimax(gameState, self.depth)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        # creates a list of tuples- (action, child)
        children = [(action,
                     gameState.generateSuccessor(self.index, action)) for action in gameState.getLegalActions(self.index)]

        max_value = float('-inf')
        action_to_max = 0  # the action that leads to max value
        alpha = float('-inf')
        beta = float('inf')
        for action, child in children:
            child.switch_turn(child.turn)
            value = self.alpha_beta(child, self.index, self.depth - 1, alpha, beta)
            if value > max_value:
                max_value = value
                action_to_max = action
        return action_to_max

    def alpha_beta(self, state, agent, depth, alpha, beta):
        if state.is_terminal() or depth == 0:
            return self.evaluationFunction(state)

        children = [(action, state.generateSuccessor(self.index, action)) for action in state.getLegalActions(self.index)]
        if state.turn == agent:
            curr_max = float('-inf')
            for action, child in children:
                child.switch_turn(child.turn)
                curr_max = max(curr_max, self.alpha_beta(child, self.index, depth - 1, alpha, beta))
                if curr_max > beta:
                    break
                else:
                    alpha = max(alpha, curr_max)
            return curr_max
        else:
            curr_min = float('inf')
            for action, child in children:
                child.switch_turn(child.turn)
                curr_min = min(curr_min, self.alpha_beta(child, self.index, depth - 1, alpha, beta))
                if curr_min < alpha:
                    break
                beta = min(beta, curr_min)
            return curr_min


class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        def expectimax(state, depth):

            curr_action = None
            if state.is_terminal() or depth == 0:
                return self.evaluationFunction(state), curr_action

            turn = state.turn
            # creates a list of tuples- (action, child)
            children = [(action, state.generateSuccessor(self.index, action)) for action in state.getLegalActions(turn)]
            if turn == 1:
                curr_max = float('-inf')
                for action, child in children:
                    child.switch_turn(child.turn)
                    value, _ = expectimax(child, depth - 1)
                    if curr_max < value:
                        curr_max = value
                        curr_action = action
                return curr_max, curr_action
            else:  # turn == 0
                value = 0
                probability = 1 / len(children)
                for action, child in children:
                    child.switch_turn(child.turn)
                    v, _ = expectimax(child, depth - 1)
                    value += v
                return value * probability, action

        _, action = expectimax(gameState, self.depth)
        return action
