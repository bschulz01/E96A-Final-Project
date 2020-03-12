# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
from typing import List, Any

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index) # starting index of the pacman
    self.numFood = len(self.getFood(gameState).asList()) # the amount of food that has not been returned
    self.hasFood = False
    self.offensiveIndex = self.getTeam(gameState)[0] # agent index of the offensive agent
    '''
    Your initialization code goes here, if you need any.
    '''

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    if len(self.getFood(gameState).asList()) < self.numFood:
      if gameState.getAgentState(self.offensiveIndex).isPacman == True:
        self.hasFood = True
      else:
        self.hasFood = False
        self.numFood = len(self.getFood(gameState).asList())
    if gameState.getAgentState(self.offensiveIndex).isPacman == False:
      self.hasFood = False

    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.getValue(gameState, a, self.index, 2) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

     # if self.index == 1:
    #   print(maxValue)

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = min([self.getMazeDistance(point, pos2) for point in self.getHomeLocations(gameState)])

        if dist < bestDist:
          bestAction = action
          bestDist = dist

      return bestAction

    return random.choice(bestActions)

  def getHomeLocations(self, gameState):
    if self.getTeam(gameState)[0] % 2 == 0:
      w = gameState.getWalls().width // 2
      locations = [(x, y) for (x, y) in gameState.getWalls().asList(False) if x == w]
      #self.debugDraw(locations, [1,0,0])
    else:
      w = gameState.getWalls().width // 2
      locations = [(x, y) for (x, y) in gameState.getWalls().asList(False) if x == w]
      #self.debugDraw(locations, [0,0,1])
    return locations

  def getValue(self, gameState, action, index, depth):

    if index == self.index:
      depth -= 1;

    if depth <= 0:
      return self.evaluate(gameState, action)

    agents = [a for a in self.getOpponents(gameState) if self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(a)) < 10]

    agents.append(self.index)

    currentIndex = 0

    for i, agent in enumerate(agents):
      if agent == index:
        currentIndex = i

    nextIndex = agents[(currentIndex + 1) % len(agents)]
    successorState = gameState.generateSuccessor(index, action)

    values = [self.getValue(successorState, action, nextIndex, depth) for action in successorState.getLegalActions(nextIndex)]

    reward = self.getReward(gameState, action, successorState)

    if index == self.index:
      return reward + 0.8 * max(values)
    else:
      return min(values)

  # see if there is only one path to go down
  def getPath(self, gameState, action, length):
    newActions = [a for a in gameState.getLegalActions if a != 'Stop' and a != self.invertAction(action)]
    if len(newActions) <= 1:
      length += 1

  def invertAction(action):
    if action == 'North':
      return 'South'
    elif action == 'South':
      return 'North'
    elif action == 'East':
      return 'West'
    elif action == 'West':
      return 'East'
    else:
      return 'Stop'

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def getReward(self, gameState, action, successorState):
    return 0

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}


class OffensiveReflexAgent(ReflexAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)
    #self.getScore(successor)

    myPos = successor.getAgentState(self.index).getPosition()

    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    # Compute proximity to nearest ghost if you're a pacman and are trying to eat food
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    nearestGhosts = [self.getMazeDistance(myPos, a.getPosition()) for a in enemies if a.isPacman == False and a.getPosition() != False]
    if len(nearestGhosts) != 0:
      nearestGhost = min(nearestGhosts)
      if nearestGhost <= 1:
        features['nearestGhost'] = 2*nearestGhost
      elif nearestGhost <= 4:
        features['nearestGhost'] = nearestGhost
      else:
        features['nearestGhost'] = 0
    else:
        features['nearestGhost'] = 0
        features['distanceToFood'] *= 2 # focus more on food if there are no ghosts


    capsules = self.getCapsules(gameState)
    if len(capsules) != 0:
      features['nearestCapsule'] = min([self.getMazeDistance(myPos, capsule) for capsule in capsules])
    else:
      features['nearestCapsule'] = 0

    features['distanceToHome'] = min([self.getMazeDistance(myPos, x) for x in self.getHomeLocations(gameState)])

    return features

  def getWeights(self, gameState, action):
    weights = {'successorScore': 50, 'distanceToFood': -1, 'nearestGhost': -5}
    # determine the weight of a capsule based on if any are left
    if len(self.getCapsules(gameState)) != 0:
      weights['nearestCapsule'] = -1
    else:
      weights['nearestCapsule'] = 0
    # Weight the distance to a ghost more heavily if you are pacman (if they can eat you)
    if (gameState.getAgentState(self.index).isPacman == True):
      weights['nearestGhost'] = -10
    # if you have food, focus on going home and avoiding ghosts
    if (self.hasFood):
      numFood = self.numFood = len(self.getFood(gameState).asList())
      weights['distanceToHome'] = -2*numFood
      weights['nearestGhost'] -= 50
    else:
      weights['distanceToHome'] = 0

    # if you are far away from home, focus more on gaining more food
    distanceToHome = min([self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), x) for x in self.getHomeLocations(gameState)])
    if distanceToHome > 5 and self.hasFood:
      weights['distanceToFood'] -= 3

    return weights

  def getReward(self, gameState, action, successorState):

    # see if it eats food
    reward = 3*(len(self.getFood(gameState).asList()) - len(self.getFood(successorState).asList()))

    if len(self.getCapsules(successorState)) < len(self.getCapsules(gameState)):
      reward += 2

    # see if we get eaten
    if successorState.getAgentPosition(self.index) == self.start:
      if (self.hasFood):
        reward -= 10
      else:
        reward -= 5

    # get a reward for returning home with food
    if self.hasFood and gameState.getAgentState(self.index).isPacman == True:
      if successorState.getAgentState(self.index).isPacman == False:
        reward += 1

    # if you are trapped
    if self.isTrapped(successorState):
      if self.hasFood:
        reward -= 5
      else:
        reward -= 1

    reward *= 50

    return reward

  def isTrapped(self, gameState):
    myPos = gameState.getAgentState(self.index).getPosition()
    enemies = [a for a in self.getOpponents(gameState) if gameState.getAgentState(a).isPacman == False]
    if len(enemies) != 0:
      closestEnemyDistance = min([self.getMazeDistance(myPos, gameState.getAgentState(a).getPosition()) for a in enemies])

      closestEnemyIndex = random.choice([a for a in enemies if self.getMazeDistance(myPos, gameState.getAgentState(a).getPosition()) == closestEnemyDistance])
      closestEnemy = gameState.getAgentState(closestEnemyIndex).getPosition()

      if len([action for action in gameState.getLegalActions(self.index) if action != 'Stop']):
        if (closestEnemy[0] == myPos[0] and (closestEnemy[1] == myPos[1] + 1 or closestEnemy[1] == myPos[1] - 1)) or (closestEnemy[1] == myPos[1] and (closestEnemy[0] == myPos[0] + 1 or closestEnemy[0] == myPos[0] - 1)):
          print("Trapped!")
          return True
    return False;


class DefensiveReflexAgent(ReflexAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """


  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)
    else: # if there are no invaders, go to the border of your side
      dists = [self.getMazeDistance(myPos, pos) for pos in self.getHomeLocations(gameState)]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    # Calculate enemy's nearest distance to food

    # see if the enemy is trapped
    if self.isTrapped(gameState):
      features['trapped'] = 10000000
    else:
      features['trapped'] = 0

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'trapped': 100}

  def getReward(self, gameState, action, successorState):
    # see if there is one fewer enemy
    newEnemies = [successorState.getAgentState(i) for i in self.getOpponents(successorState)]
    oldEnemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

    reward = len(oldEnemies) - len(newEnemies)

    if self.isTrapped(successorState):
      reward += 3

    reward *= 50

    return reward

  def isTrapped(self, gameState):
    myPos = gameState.getAgentState(self.index).getPosition()
    enemies = [a for a in self.getOpponents(gameState) if gameState.getAgentState(a).isPacman]
    if len(enemies) != 0:
      closestEnemyDistance = min([self.getMazeDistance(myPos, gameState.getAgentState(a).getPosition()) for a in enemies])

      closestEnemy = random.choice([a for a in enemies if self.getMazeDistance(myPos, gameState.getAgentState(a).getPosition()) == closestEnemyDistance])

      enemyActions = [action for action in gameState.getLegalActions(closestEnemy) if action != 'Stop']

      if len(enemyActions) == 1:
        enemyPosition = gameState.getAgentState(closestEnemy).getPosition()
        if (enemyPosition[0] == myPos[0] and (enemyPosition[1] == myPos[1] + 1 or enemyPosition[1] == myPos[1] - 1)) or (enemyPosition[1] == myPos[1] and (enemyPosition[0] == myPos[0] + 1 or enemyPosition[0] == myPos[0] - 1)):
          return True
    return False;


