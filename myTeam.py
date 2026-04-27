# myTeam.py
# Name: Aditya Singh
# NetID: ars230009
# Class: CS 4365.HONORS

from captureAgents import CaptureAgent
import random, time, util
from util import nearestPoint, PriorityQueue
from game import Directions
import game

# 
# Important Adjustable constants
# 

GHOST_DANGER_DIST = 5   # How close (maze steps) a ghost must be to be "scary"
CARRY_LIMIT = 5   # Return home after collecting this many pellets
TIME_PRESSURE = 100 # Remaining moves at which we force a return-home
PATROL_DEPTH = 4   # How far from the border the defender patrols


#
# Team Creation 
#

def createTeam(firstIndex, secondIndex, isRed, first='Oofius', second='Doofius'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


#
# Base Agent
#

class BaseAgent(CaptureAgent):
    """
    Provides helper methods that both my agents need to make decisions and navigate the maze.
    """

    # Initialization and state tracking
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # Where we respawn after being eaten
        self.start = gameState.getAgentPosition(self.index)
        # Positions along the border line that we can actually walk through
        self.boundaryPositions = self._buildBoundaryPositions(gameState)

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Halfway between two grid cells — advance to the next cell
            return successor.generateSuccessor(self.index, action)
        return successor

    # Maze navigation
    def aStar(self, gameState, goalPositions):
        """
        A* search from my agent's current position to the nearest goal.
        Returns the first action to take, or Directions.STOP if already there
        or no path exists.

        goalPositions : list of (x, y) tuples we want to reach
        """
        startPosition = gameState.getAgentState(self.index).getPosition()

        if not goalPositions:
            return Directions.STOP

        # Convert goal list to a set for O(1) look-up
        goals = set(goalPositions)

        # Priority queue entries: (f, g, position, first_action)
        # f = g + h  (total estimated cost)
        # g = cost so far (number of steps taken)
        # h = heuristic (maze distance to nearest goal)
        priorityQueue = PriorityQueue()
        visited = set()

        heuristicDistance = min(self.getMazeDistance(startPosition, g) for g in goals)
        priorityQueue.push((startPosition, None, 0), heuristicDistance)

        while not priorityQueue.isEmpty():
            pos, firstAction, g = priorityQueue.pop()
            if pos in visited:
                continue
            visited.add(pos)
            # Goal check
            if pos in goals:
                if firstAction is None:
                    return Directions.STOP # Already reached a goal
                else:
                    return firstAction

            # Expand neighbours (all legal moves from pos)
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nextx, nexty = int(pos[0]+dx), int(pos[1]+dy)
                nextpos = (nextx, nexty)
                if nextpos in visited:
                    continue
                if gameState.hasWall(nextx, nexty):
                    continue
                newG = g + 1
                heuristicDistance = min(self.getMazeDistance(nextpos, goal) for goal in goals)
                if firstAction is None:
                    action = self._deltaToDirection(dx, dy)
                else:
                    action = firstAction
                priorityQueue.push((nextpos, action, newG), newG + heuristicDistance)
        # No path found
        return Directions.STOP

    def _deltaToDirection(self, dx, dy):
        """Converts a (dx, dy) movement delta into a Directions constant."""
        if dx ==  1: return Directions.EAST
        if dx == -1: return Directions.WEST
        if dy ==  1: return Directions.NORTH
        if dy == -1: return Directions.SOUTH
        return Directions.STOP

    # Senses Enemies Methods
    def getVisibleEnemies(self, gameState):
        """
        Returns a list of AgentState objects for opponents that my agents can
        currently see (position is not None).
        """
        return [
            gameState.getAgentState(i)
            for i in self.getOpponents(gameState)
            if gameState.getAgentState(i).getPosition() is not None
        ]

    def getThreateningGhosts(self, gameState):
        """
        Returns enemy ghost AgentStates that are:
          - Visible (position known)
          - On their own side (not a Pacman)
          - NOT scared (or scared for 2 or fewer moves)
        These are the ghosts that can actually eat my agents, so avoid them.
        """
        return [
            e for e in self.getVisibleEnemies(gameState)
            if not e.isPacman and e.scaredTimer <= 2
        ]

    def getScaredGhosts(self, gameState):
        """
        Returns enemy ghosts that are currently scared and can be eaten.
        Useful for my offensive agent to go hunting instead of fleeing.
        """
        return [
            e for e in self.getVisibleEnemies(gameState)
            if not e.isPacman and e.scaredTimer > 2
        ]

    def getVisibleInvaders(self, gameState):
        """
        Returns enemy Pacmen that have crossed into our territory and are
        visible.  My defensive agent chases them.
        """
        return [
            e for e in self.getVisibleEnemies(gameState)
            if e.isPacman
        ]

    # Boundary and distance helpers
    def _buildBoundaryPositions(self, gameState):
        """
        Pre-computes all walkable cells on my side of the centre line.
        I keep only the column right at the border so it doubles as a
        return-home target for my offensive agent.
        """
        layout = gameState.data.layout
        width, height = layout.width, layout.height
        mid = width // 2

        # The column index at the border edge
        col = mid - 1 if self.red else mid

        return [
            (col, y)
            for y in range(height)
            if not gameState.hasWall(col, y)
        ]

    def distanceToHome(self, gameState):
        """
        Shortest maze distance from my agent's current position to the
        nearest boundary cell. Useful to make sure agents know how far they are from home when fleeing or returning.
        """
        myPos = gameState.getAgentState(self.index).getPosition()
        if not self.boundaryPositions:
            return 0
        return min(self.getMazeDistance(myPos, b) for b in self.boundaryPositions)

    def evaluate(self, gameState, action):
        """Standard linear feature-weight evaluation used by both agents."""
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    # Subclasses must implement these two methods
    def getFeatures(self, gameState, action):
        raise NotImplementedError
    def getWeights(self, gameState, action):
        raise NotImplementedError


# 
# Offensive Agent
# 

class Oofius(BaseAgent):
    """
    Crosses the centre line, collects food, and returns home to score.

    Decision priority (highest to lowest):
      1. If a non-scared ghost is within GHOST_DANGER_DIST steps → run away / use capsule
      2. If carrying >= CARRY_LIMIT, or time is short, or food almost gone → go home
      3. If a scared ghost is nearby → eat it (free kill + time advantage)
      4. Otherwise → eat the nearest food pellet
    """

    def registerInitialState(self, gameState):
        BaseAgent.registerInitialState(self, gameState)
        # We'll use this to detect when we've just respawned
        self.lastPos = self.start

    # Choose action
    def chooseAction(self, gameState):
        """
        Main decision function called every turn.
        We use A* for movement and a feature evaluator to choose which goal
        to pursue.
        """
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        carrying = myState.numCarrying
        timeLeft = gameState.data.timeleft
        foodLeft = len(self.getFood(gameState).asList())

        threats = self.getThreateningGhosts(gameState)
        scaredGhosts = self.getScaredGhosts(gameState)
        capsules = self.getCapsules(gameState)

        # Run away if a threatening ghost is dangerously close
        if threats:
            closestGhostDist = min(
                self.getMazeDistance(myPos, g.getPosition()) for g in threats
            )
            if closestGhostDist <= GHOST_DANGER_DIST:
                # If there is a capsule reachable before the ghost catches my agent, go for it
                if capsules:
                    bestCapsule = min(
                        capsules, key=lambda c: self.getMazeDistance(myPos, c)
                    )
                    capDist = self.getMazeDistance(myPos, bestCapsule)
                    if capDist < closestGhostDist:
                        return self.aStar(gameState, [bestCapsule])

                # Otherwise retreat to my side immediately
                return self.aStar(gameState, self.boundaryPositions)

        # Return home if we have enough food or time is running out 
        shouldReturn = (
            carrying >= CARRY_LIMIT  
            or foodLeft <= 2     
            or (carrying > 0 and timeLeft < TIME_PRESSURE)
        )
        if shouldReturn:
            return self.aStar(gameState, self.boundaryPositions)

        # Eat scared ghosts for an advantage
        if scaredGhosts:
            ghostPositions = [g.getPosition() for g in scaredGhosts]
            return self.aStar(gameState, ghostPositions)

        # Go get food
        foodList = self.getFood(gameState).asList()
        if foodList:
            targetFood = self._chooseSafeFood(gameState, foodList, threats)
            return self.aStar(gameState, [targetFood])

        # Fallback: nothing useful to do
        return Directions.STOP

    # Selecting best food
    def _chooseSafeFood(self, gameState, foodList, threats):
        """
        Picks which food pellet to go for.
        My Strategy:
          • If there are no visible threats, just pick the nearest pellet.
          • If there are visible threats, score each pellet by
              score = -distToFood  +  (safetyBonus * distToNearestGhost)
            so we prefer food that is both close to us and far from ghosts.
        """
        myPos = gameState.getAgentState(self.index).getPosition()

        if not threats:
            # Go to the nearest food pellet
            return min(foodList, key=lambda f: self.getMazeDistance(myPos, f))

        # Weighted safety score
        bestFood, bestScore = None, float('-inf')
        for food in foodList:
            distToFood  = self.getMazeDistance(myPos, food)
            ghostDists  = [self.getMazeDistance(food, g.getPosition()) for g in threats]
            minGhostDist = min(ghostDists)

            # Higher score = more desirable
            # We subtract dist to food (closer is better) and add ghost dist
            # (farther from ghost is safer). The 0.5 weight lets us still
            # prefer food that is a bit closer even if slightly less safe.
            score = -distToFood + 0.5 * minGhostDist
            if score > bestScore:
                bestScore = score
                bestFood  = food

        return bestFood

    # Fallback feature evalutaor (used for tiebreaking)
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        foodList = self.getFood(successor).asList()
        features['foodRemaining'] = -len(foodList)
        if foodList:
            features['distToClosestFood'] = min(
                self.getMazeDistance(myPos, f) for f in foodList)

        threats = self.getThreateningGhosts(successor)
        if threats:
            dists = [self.getMazeDistance(myPos, g.getPosition()) for g in threats]
            features['ghostDanger'] = 1.0 / max(min(dists), 0.5)

        features['distHome'] = self.distanceToHome(successor)
        features['carrying'] = myState.numCarrying

        return features

    def getWeights(self, gameState, action):
        return {
            'foodRemaining':    100,
            'distToClosestFood': -2,
            'ghostDanger':      -500,
            'distHome':         -0.5,
            'carrying':          1,
        }


# 
# Defensive Agent
# 

class Doofius(BaseAgent):
    """
    Stays on my side and prevents the enemy from scoring.

    Behaviour modes (in priority order):
      1. Chase a visible invader
      2. If we are scared (just got hit by a power capsule), keep distance
         from invaders rather than charging them
      3. If no invader is visible but we saw one recently, go to where
         the food was last eaten (likely still nearby)
      4. Patrol between two fixed points near the border to intercept crossings
    """

    def registerInitialState(self, gameState):
        BaseAgent.registerInitialState(self, gameState)

        # Build a patrol route: two points that bracket the vertical centre of
        # the map along our border column, separated by PATROL_DEPTH rows.
        # The agent will oscillate between them when idle.
        self.patrolPoints = self._buildPatrolPoints(gameState)
        self.patrolIndex  = 0   # which patrol point we're heading toward

        # Track which food pellets we had last turn so we can detect when an
        # invader eats one (even if we can't see the invader directly).
        self.prevFoodCount   = len(self.getFoodYouAreDefending(gameState).asList())
        self.lastEatenTarget = None   # position of most recently eaten food

    # Choose action
    def chooseAction(self, gameState):
        """
        Main decision function called every turn.
        """
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        isScared = myState.scaredTimer > 0

        invaders = self.getVisibleInvaders(gameState)

        # Detect if an enemy just ate one of my food pellets this turn
        self._updateEatenFood(gameState)

        # If Scared: keep distance from invaders to avoid giving them points
        if isScared and invaders:
            return self._fleeFromInvaders(gameState, invaders)

        # Chase visible invaders
        if invaders:
            return self._chaseInvader(gameState, invaders)

        # Go to last known eaten-food position
        if self.lastEatenTarget is not None:
            dist = self.getMazeDistance(myPos, self.lastEatenTarget)
            if dist > 0:
                return self.aStar(gameState, [self.lastEatenTarget])
            else:
                # My agent arrived — clear the target, resume patrol
                self.lastEatenTarget = None

        # Stay close to the border
        return self._patrol(gameState)

    # Chasing invader
    def _chaseInvader(self, gameState, invaders):
        """
        Heads directly for the nearest visible invader using A*.
        """
        myPos = gameState.getAgentState(self.index).getPosition()
        # Target the closest invader
        target = min(invaders, key=lambda e: self.getMazeDistance(myPos, e.getPosition()))
        return self.aStar(gameState, [target.getPosition()])

    # Running away from invaders when scared
    def _fleeFromInvaders(self, gameState, invaders):
        """
        When my agents are scared, move AWAY from invaders to avoid being eaten.
        We pick the legal action that maximises distance to the nearest invader.
        """
        myPos = gameState.getAgentState(self.index).getPosition()
        actions = gameState.getLegalActions(self.index)
        actions = [a for a in actions if a != Directions.STOP] or actions

        bestAction, bestDist = None, -1
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            pos       = successor.getAgentState(self.index).getPosition()
            minDist   = min(self.getMazeDistance(pos, e.getPosition()) for e in invaders)
            if minDist > bestDist:
                bestDist   = minDist
                bestAction = action

        return bestAction or random.choice(actions)

    # Patrolling logic
    def _patrol(self, gameState):
        """
        Move toward the current patrol waypoint.  When my agent arrives, advance to
        the next waypoint (cycling through the list).
        """
        myPos  = gameState.getAgentState(self.index).getPosition()
        target = self.patrolPoints[self.patrolIndex]

        if myPos == target:
            # Arrived, pick next waypoint
            self.patrolIndex = (self.patrolIndex + 1) % len(self.patrolPoints)
            target = self.patrolPoints[self.patrolIndex]

        return self.aStar(gameState, [target])

    # Figures out patrol locations based on map layout
    def _buildPatrolPoints(self, gameState):
        """
        Generates a short list of patrol waypoints along the border column
        spread across the vertical centre of the map. Two points works well for most maps.
        """
        layout = gameState.data.layout
        height = layout.height
        mid_y = height // 2

        # I want points near the vertical centre, separated by PATROL_DEPTH
        candidates = []
        for dy in range(-PATROL_DEPTH, PATROL_DEPTH + 1):
            y = mid_y + dy
            if 0 < y < height:
                candidates.append(y)

        # Filter to walkable cells on the boundary column
        waypoints = []
        for b in self.boundaryPositions:
            if b[1] in candidates:
                waypoints.append(b)

        # If my agent somehow found nothing, fall back to all boundary positions
        if not waypoints:
            waypoints = self.boundaryPositions

        # Keep two points: upper-centre and lower-centre of the candidates so the agent moves a lot and covers more vertical ground
        waypoints.sort(key=lambda p: p[1])
        if len(waypoints) >= 2:
            return [waypoints[0], waypoints[-1]]
        return waypoints

    # Food-eaten tracking (for defense)
    def _updateEatenFood(self, gameState):
        """
        Compares how many food pellets we had last turn versus now.
        If a pellet disappeared, it means an invader ate it — I store its
        position as a soft target so my defender can investigate even without
        a direct line of sight.
        """
        currentFood  = self.getFoodYouAreDefending(gameState).asList()
        currentCount = len(currentFood)

        if currentCount < self.prevFoodCount:
            # At least one pellet was eaten this turn. Find which cells disappeared by comparing to last turn's food.
            prevFood = self.getFoodYouAreDefending(
                self.getPreviousObservation()
            ).asList() if self.getPreviousObservation() else currentFood

            eaten = [f for f in prevFood if f not in currentFood]
            if eaten:
                # Target the eaten pellet closest to my agent's current position
                myPos = gameState.getAgentState(self.index).getPosition()
                self.lastEatenTarget = min(
                    eaten, key=lambda f: self.getMazeDistance(myPos, f)
                )

        self.prevFoodCount = currentCount

    # Fallback feature evaluator
    def getFeatures(self, gameState, action):
        features  = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState   = successor.getAgentState(self.index)
        myPos     = myState.getPosition()

        features['onDefense'] = 0 if myState.isPacman else 1

        invaders = self.getVisibleInvaders(successor)
        features['numInvaders'] = len(invaders)
        if invaders:
            features['invaderDist'] = min(
                self.getMazeDistance(myPos, e.getPosition()) for e in invaders)

        if action == Directions.STOP:
            features['stop'] = 1

        rev = Directions.REVERSE[
            gameState.getAgentState(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    # Uses arbitrary weights, I tried a few different values to find a decent balance but this is by no means optimal
    def getWeights(self, gameState, action):
        return {
            'onDefense': 100,
            'numInvaders': -1000,
            'invaderDist': -10,
            'stop': -100,
            'reverse': -2,
        }
