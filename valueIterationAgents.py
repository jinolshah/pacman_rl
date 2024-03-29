# valueIterationAgents.pyupdatedValues
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            states = self.mdp.getStates()
            updatedValues = {}

            for state in states:
                maxval = 0
                actions = self.mdp.getPossibleActions(state)
                
                if actions:
                    maxval = max([self.computeQValueFromValues(state, action) for action in actions])

                updatedValues[state] = maxval

            for state in states:
                self.values[state] = updatedValues[state]



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qval = 0

        for nextState, problem in self.mdp.getTransitionStatesAndProbs(state, action):
            qval += problem*(self.mdp.getReward(state, action, nextState) + self.discount*self.getValue(nextState))

        return qval

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        
        values = [self.computeQValueFromValues(state, action) for action in actions]
        return actions[values.index(max(values))] if actions else None

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        numStates = len(states)
        
        for i in range(self.iterations):
            state = states[i%numStates]
            actions = self.mdp.getPossibleActions(state)

            maxval = 0
            if actions:
                maxval = max([self.computeQValueFromValues(state, action) for action in actions])

            self.values[state] = maxval

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        prev_states = {}
        states = self.mdp.getStates()

        for s in states:
            prev_states[s] = set()

        for s in states:
            for a in self.mdp.getPossibleActions(s):
                statesAndProbs = self.mdp.getTransitionStatesAndProbs(s, a)
                for state, prob in statesAndProbs:
                    prev_states[state].add(s)

        pq = util.PriorityQueue()

        for s in states:
            if self.mdp.isTerminal(s):
                continue
            best_a = self.computeActionFromValues(s)
            highest_q = self.computeQValueFromValues(s, best_a)
            difference = abs(highest_q - self.values[s])
            pq.push(s, -difference)

        for _ in range(self.iterations):
            if pq.isEmpty():
                return
            state = pq.pop()

            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                max_value = float('-infinity')
                for action in actions:
                    val = 0
                    next_state_infos = self.mdp.getTransitionStatesAndProbs(state, action)
                    for next_state_info in next_state_infos:
                        next_state = next_state_info[0]
                        prob = next_state_info[1]
                        reward = self.mdp.getReward(state, action, next_state)
                        v = self.values[next_state]
                        val += prob * (reward + self.discount * v)
                    max_value = max(max_value, val)
                if max_value > float('-infinity'):
                    self.values[state] = max_value
                else:
                    self.values[state] = 0

            for p_state in prev_states[state]:
                best_a = self.computeActionFromValues(p_state)
                if best_a == None:
                    continue
                highest_q = self.computeQValueFromValues(p_state, best_a)
                difference = abs(highest_q - self.values[p_state])

                if difference > self.theta:
                    pq.update(p_state, -difference)
