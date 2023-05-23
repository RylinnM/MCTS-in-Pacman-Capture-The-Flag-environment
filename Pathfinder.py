# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# -----------------------------------------------------------------------------
# Written by: Ruilin Ma, 2023
# Version: 1.0
# Date: 2023/05/22

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import numpy as np

import math


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


max_depth = 15


#####################
# Structure for UCT #
#####################
class MCTSNode(object):

    def __init__(self, gameState, agent, action, parent, enemy_pos, borderline):
        self.parent = parent
        self.action = action
        self.depth = parent.depth + 1 if parent else 0

        self.child = []
        self.visits = 1
        self.q_value = 0.0

        self.gameState = gameState.deepCopy()
        self.enemy_pos = enemy_pos
        self.legalActions = [act for act in gameState.getLegalActions(agent.index) if act != 'Stop']
        self.unexploredActions = self.legalActions[:]
        self.borderline = borderline

        self.agent = agent
        self.epsilon = 1
        self.rewards = 0

    def node_expanded(self):
        if self.depth >= max_depth:
            return self

        if self.unexploredActions != []:
            action = self.unexploredActions.pop()
            current_game_state = self.gameState.deepCopy()
            next_game_state = current_game_state.generateSuccessor(self.agent.index, action)
            child_node = MCTSNode(next_game_state, self.agent, action, self, self.enemy_pos, self.borderline)
            self.child.append(child_node)
            return child_node

        if util.flipCoin(self.epsilon):
            next_best_node = self.sel_best_child()
        else:
            next_best_node = random.choice(self.child)
        return next_best_node.node_expanded()

    def mcts_search(self):
        timeLimit = 0.99
        start = time.time()
        while (time.time() - start < timeLimit):

            node_selected = self.node_expanded()

            reward = node_selected.cal_reward()

            node_selected.backpropagation(reward)

        return self.sel_best_child().action

    def sel_best_child(self):
        best_score = -np.inf
        best_child = None
        for candidate in self.child:
            score = candidate.q_value / candidate.visits
            if score > best_score:
                best_score = score
                best_child = candidate
        return best_child

    def backpropagation(self, reward):
        self.visits += 1
        self.q_value += reward
        if self.parent is not None:
            self.parent.backpropagation(reward)

    def cal_reward(self):
        current_pos = self.gameState.getAgentPosition(self.agent.index)
        if current_pos == self.gameState.getInitialAgentPosition(self.agent.index):
            return -1000
        value = self.get_feature() * MCTSNode.get_weight_backup(self)
        return value

    def get_feature(self):
        gameState = self.gameState
        feature = util.Counter()
        current_pos = self.gameState.getAgentPosition(self.agent.index)
        feature['distance'] = min([self.agent.getMazeDistance(current_pos, borderPos) for borderPos in self.borderline])
        return feature

    def get_weight(self):
        return {'minDistToFood': -10, 'getFood': 100}

    def get_weight_backup(self):
        return {'distance': -1}

# --------------------------------------------------------------------------

##########
# Agents #
##########

class OffensiveAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.food_count = int(len(self.getFood(gameState).asList()))
        self.arena_width = gameState.data.layout.width
        self.arena_height = gameState.data.layout.height
        self.friendly_borders = self.detect_my_border(gameState)
        self.hostile_borders = self.detect_enemy_border(gameState)

    def detect_my_border(self, gameState):
        """
        Return borders position
        """
        walls = gameState.getWalls().asList()
        if self.red:
            border_x = self.arena_width // 2 - 1
        else:
            border_x = self.arena_width // 2
        border_line = [(border_x, h) for h in range(self.arena_height)]
        return [(x, y) for (x, y) in border_line if (x, y) not in walls and (x + 1 - 2*self.red, y) not in walls]

    def detect_enemy_border(self, gameState):
        """
        Return borders position
        """
        walls = gameState.getWalls().asList()
        if self.red:
            border_x = self.arena_width // 2
        else:
            border_x = self.arena_width // 2 - 1
        border_line = [(border_x, h) for h in range(self.arena_height)]
        return [(x, y) for (x, y) in border_line if (x, y) not in walls and (x + 1 - 2*self.red, y) not in walls]

    def detect_enemy_ghost(self, gameState):
        """
        Return Observable Oppo-Ghost Index
        """
        enemyList = []
        for enemy in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(enemy)
            if (not enemyState.isPacman) and enemyState.scaredTimer == 0:
                enemyPos = gameState.getAgentPosition(enemy)
                if enemyPos != None:
                    enemyList.append(enemy)
        return enemyList

    def detect_enemy_approaching(self, gameState):
        """
        Return Observable Oppo-Ghost Position Within 5 Steps
        """
        dangerGhosts = []
        ghosts = self.detect_enemy_ghost(gameState)
        myPos = gameState.getAgentPosition(self.index)
        for g in ghosts:
            distance = self.getMazeDistance(myPos, gameState.getAgentPosition(g))
            if distance <= 5:
                dangerGhosts.append(g)
        return dangerGhosts

    def detect_enemy_pacman(self, gameState):
        """
        Return Observable Oppo-Pacman Position
        """
        enemyList = []
        for enemy in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(enemy)
            if enemyState.isPacman and gameState.getAgentPosition(enemy) != None:
                enemyList.append(enemy)
        return enemyList

    def chooseAction(self, gameState):
        """
        Picks best actions.
        """
        start = time.time()
        actions = gameState.getLegalActions(self.index)
        agent_state = gameState.getAgentState(self.index)

        carrying = agent_state.numCarrying
        isPacman = agent_state.isPacman

        if isPacman:
            appr_ghost_pos = [gameState.getAgentPosition(g) for g in self.detect_enemy_approaching(gameState)]
            foodList = self.getFood(gameState).asList()

            if not appr_ghost_pos:
                values = [self.evaluate_off(gameState, a) for a in actions]
                maxValue = max(values)
                bestActions = [a for a, v in zip(actions, values) if v == maxValue]
                action_chosen = random.choice(bestActions)

            elif len(foodList) < 2 or carrying > 7:
                rootNode = MCTSNode(gameState, self, None, None, appr_ghost_pos, self.friendly_borders)
                action_chosen = MCTSNode.mcts_search(rootNode)
            else:
                rootNode = MCTSNode(gameState, self, None, None, appr_ghost_pos, self.friendly_borders)
                action_chosen = MCTSNode.mcts_search(rootNode)

        else:
            ghosts = self.detect_enemy_ghost(gameState)
            values = [self.evaluate_def(gameState, a, ghosts) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            action_chosen = random.choice(bestActions)

        return action_chosen

    def evaluate_off(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_off_features(gameState, action)
        weights = self.get_off_weights(gameState, action)
        return features * weights

    def get_off_features(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        next_tate = self.get_next_state(gameState, action)
        if next_tate.getAgentState(self.index).numCarrying > gameState.getAgentState(self.index).numCarrying:
            features['getFood'] = 1
        else:
            if len(self.getFood(next_tate).asList()) > 0:
                features['minDistToFood'] = self.get_min_dist_to_food(next_tate)
        return features

    def get_off_weights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'minDistToFood': -1, 'getFood': 100}

    def evaluate_def(self, gameState, action, ghosts):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_def_features(gameState, action)
        weights = self.get_def_weights(gameState, action)
        return features * weights

    def get_def_features(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_next_state(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            current_pos = successor.getAgentState(self.index).getPosition()
            min_distance = min([self.getMazeDistance(current_pos, food) for food in foodList])
            features['distanceToFood'] = min_distance
        return features

    def get_def_weights(self, gameState, action):

        return {'successorScore': 100, 'distanceToFood': -1}

    def get_next_state(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor

    def get_min_dist_to_food(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        return min([self.getMazeDistance(myPos, f) for f in self.getFood(gameState).asList()])


class DefensiveAgent(OffensiveAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_def_features(self, gameState, action):
        features = util.Counter()
        next_state = self.get_next_state(gameState, action)

        my_state = next_state.getAgentState(self.index)
        my_pos = my_state.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if my_state.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [next_state.getAgentState(i) for i in self.getOpponents(next_state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(my_pos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_def_weights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

