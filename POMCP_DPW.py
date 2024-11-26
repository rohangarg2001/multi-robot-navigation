import numpy as np
from collections import defaultdict
import math
import random

class History:
    def __init__(self):
        self.actions = []
        self.observations = []

    def add_step(self, action, observation):
        self.actions.append(action)
        self.observations.append(observation)

    def get_sequence(self):
        return list(zip(self.actions, self.observations))

    def __str__(self):
        return str(self.get_sequence())

    def copy(self):
        new_history = History()
        new_history.actions = self.actions.copy()
        new_history.observations = self.observations.copy()
        return new_history


class Node:
    def __init__(self, history, parent=None):
        self.history = history  # History object representing the path to this node
        self.parent = parent    # Parent Node
        self.children = dict()  # Maps actions to child Nodes
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = self.get_possible_actions()

    def get_possible_actions(self):
        """
        Return a list of possible actions.
        For continuous actions, you might need to sample or discretize.
        """
        # Placeholder: Implement based on your action space
        return sample_continuous_actions()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        """
        Select the child with the highest UCB1 value.
        """
        choices_weights = [
            (child.total_reward / child.visits) +
            c_param * math.sqrt((2 * math.log(self.visits) ) / child.visits)
            for child in self.children.values()
        ]
        return list(self.children.values())[np.argmax(choices_weights)]

    def expand(self):
        """
        Expand by adding a new child node for an untried action.
        """
        action = self.untried_actions.pop()
        new_history = self.history.copy()
        # Note: Observation is not known yet; it will be determined during simulation
        child_node = Node(new_history, parent=self)
        self.children[action] = child_node
        return action, child_node

    def update(self, reward):
        """
        Update node statistics.
        """
        self.visits += 1
        self.total_reward += reward


class Tree:
    def __init__(self, root_state, domain):
        self.root = Node(History())
        self.domain = domain  # Domain-specific object handling dynamics

    def search(self, num_simulations):
        for _ in range(num_simulations):
            node = self.root
            history = self.root.history.copy()

            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
                action = self.get_action_from_parent(node)
                observation = self.domain.sample_observation(action, history)
                history.add_step(action, observation)

            # Expansion
            if not node.is_fully_expanded():
                action, child = node.expand()
                # Simulate observation based on action
                observation = self.domain.sample_observation(action, history)
                history.add_step(action, observation)
                node = child

            # Simulation (Rollout)
            reward = self.rollout(history)

            # Backpropagation
            while node is not None:
                node.update(reward)
                node = node.parent

    def get_action_from_parent(self, node):
        """
        Retrieve the action that led to the given node from its parent.
        """
        parent = node.parent
        for action, child in parent.children.items():
            if child == node:
                return action
        return None

    def rollout(self, history, depth=10):
        """
        Perform a rollout (simulation) from the current history.
        """
        total_reward = 0.0
        current_history = history.copy()
        for _ in range(depth):
            action = self.domain.sample_action(current_history)
            observation, reward, done = self.domain.step(action, current_history)
            current_history.add_step(action, observation)
            total_reward += reward
            if done:
                break
        return total_reward

    def best_action(self):
        """
        Return the action with the highest visit count from the root.
        """
        if not self.root.children:
            return None
        return max(self.root.children.items(), key=lambda item: item[1].visits)[0]

    def reset_root(self, action, observation):
        """
        After taking an action and receiving an observation, update the root.
        """
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.history.add_step(action, observation)
            self.root.parent = None
        else:
            self.root = Node(History())

    def sample_continuous_actions(self):
        """
        Sample a set of continuous actions.
        Adjust the sampling strategy based on your action space.
        """
        # Placeholder: Implement your action sampling logic
        return self.domain.sample_action_space()

    def sample_continuous_observations(self, action):
        """
        Sample observations based on the action.
        """
        # Placeholder: Implement your observation sampling logic
        return self.domain.sample_observation(action)
