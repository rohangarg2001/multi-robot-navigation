import random
import numpy as np
from collections import defaultdict
from generativemodel import Environment, Robot
from graphviz import Digraph
import matplotlib.pyplot as plt
# Constants

target_goals = np.array([7, 5, 5, 7, 2, 7]) 
num_robots = 3
epsilon = 1e-7
gamma = 0.95  # Discount factor
alpha_a = 0.5  # Action progressive widening parameter
alpha_o = 0.5  # Observation progressive widening parameter
k_a = 5       # Threshold for action widening
k_o = 3       # Threshold for observation widening
c_exploration = 1  # Exploration constant for UCB


'''

Number of childs
q values and associated actions
rollout policy
reward function
reduce depth

'''


class Node:
    """Class representing a single node in the history tree."""
    def __init__(self, observation=None, action=None, parent=None, is_observation_node=False):
        self.observation = observation  # The observation that led to this node
        self.action = action  # The action taken at this node
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes (actions or observations)
        self.obs_map = defaultdict(int)  # Maps observations to their counts
        self.transition_states = []  # List of sampled transition states
        self.visit_count = 0  # Number of visits to this node
        self.q_value = 0  # Q-value for this node
        self.is_observation_node = is_observation_node  # Indicates if it's an observation node

    def add_child(self, action=None, observation=None, is_observation_node=False):
        """Adds a new child node (either an action or observation)."""
        child_node = Node(action=action, observation=observation, parent=self, is_observation_node=is_observation_node)
        self.children.append(child_node)
        return child_node


class HistoryTree:
    """Class representing the history tree."""
    def __init__(self):
        self.root = Node()  # Root node corresponds to an empty history

# Placeholder for the environment transition model
def G(s, a):
    """Simulates the environment transition given state `s` and action `a`."""
    # Reshape the state `s` into a 3xN matrix where each column represents a robot's state [x, y, theta]
    states = s.reshape(-1, 3)
    
    # Apply the actions to each robot and update their states
    actions = a.reshape(-1, 2)  # Reshape action into a 2xN matrix for each robot's (v, w)
    env = Environment(n_robots= num_robots, state= s)
    for i, robot in enumerate(env.robots):
        v, w = actions[i]
        robot.x, robot.y, robot.theta = states[i]
        robot.update(v, w, env.deltaT, env.sigma_dyn)
    
    # Get the next state of all robots
    next_states = env.get_states()
    
    # Get the observations after the update
    observations = env.get_observations()
    
    # Reward calculation (example: reward for reaching the goal)
    reward = R(env)
    # Return next state, observation, and reward
    return next_states, observations, reward



def ROLLOUT(s, h_node, d):
    """
    Performs a policy rollout using computed actions to move robots toward their respective goals.
    Adds child nodes for each action and observation.

    Args:
    s: Initial state of the system (3n*1 vector).
    h_node: The current history node to which child nodes will be added.
    d: Horizon length for the rollout.

    Returns:
    total_reward: Total accumulated reward over the rollout.
    """
    total_reward = 0
    Env = Environment(n_robots=num_robots, state=s)
    Env.robots = [Robot(*s[3 * i: 3 * i + 3].flatten()) for i in range(Env.n_robots)]  # Initialize robots

    for _ in range(d):
        # Compute actions to move toward the target goals
        actions = compute_actions(Env, target_goals)

        # Add the action as a child node of the current history node
        action_node = h_node.add_child(action=actions)

        # Get the next state and observation
        Env.step(actions.reshape(-1, 2))
        reward = R(Env)
        next_state = Env.get_states()
        observations = Env.get_observations()

        # Add the observation as a child node
        for obs in observations:
            obs_node = action_node.add_child(observation=obs, is_observation_node=True)

        # Accumulate the reward
        total_reward += np.sum(reward)

        # Set the observation node as the new current node for the next step
        h_node = obs_node  # Move to the observation node for the next action

    return total_reward


# Action Progressive Widening
def ACTION_PROG_WIDEN(h_node):
    """Performs action progressive widening."""
    Bound_a = 65  #k_a * (h_node.visit_count ** alpha_a)

    if len(h_node.children) <= Bound_a:
        # Generate and add a new action
        '''
        Implemented the function below
        '''
        new_action = sample_action()  
        new_node = h_node.add_child(new_action)
        
    # Return the action with the highest Q-value (UCB)
    return max(
        h_node.children,
        key=lambda child: child.q_value +
                            c_exploration * np.sqrt(np.log(h_node.visit_count) / (epsilon + child.visit_count))
                )


def SIMULATE(s, h_node, d):
    """Simulates the action-observation tree to calculate Q-values."""
    if d == 0:
        return 0

    # Step 1: Action Progressive Widening
    a_node = ACTION_PROG_WIDEN(h_node)
    action = a_node.action

    # Step 2: Observation Progressive Widening
    Bound_obs = 50  #  k_o * (a_node.visit_count ** alpha_o)
    if len(a_node.obs_map) <= Bound_obs:
        # Sample a new observation and expand the tree
        s_next, new_obs, r = G(s, action)
        a_node.obs_map[tuple(new_obs)] += 1
        a_node.transition_states.append(s_next)

        # Create observation node
        obs_node = a_node.add_child(observation=new_obs, is_observation_node=True)

        if len(a_node.transition_states) == 1:
            
            total = r + gamma * ROLLOUT(s_next, obs_node, d - 1)
        else:
           
            total = r + gamma * SIMULATE(s_next, obs_node, d - 1)
    else:
        # Select an existing observation with weighted probabilities
        observations = list(a_node.obs_map.keys())
        counts = list(a_node.obs_map.values())
        selected_obs = random.choices(observations, weights=counts, k=1)[0]

        # Find the corresponding observation node for the selected observation
        obs_node = next(child for child in a_node.children if np.array_equal(child.observation, selected_obs))


        # Select a transition state from the node's states
        s_next = random.choice(a_node.transition_states)

        # Reward(s,a,s')
        env = Environment(n_robots= num_robots, state= s_next) 
        r = R(env)
        # Pass the observation node to SIMULATE
        total = r + gamma * SIMULATE(s_next, obs_node, d - 1)

    # Update statistics
    h_node.visit_count += 1
    a_node.visit_count += 1
    obs_node.visit_count +=1
    a_node.q_value = (a_node.q_value * (a_node.visit_count - 1) + total) / a_node.visit_count
    return total


def PLAN(b, P, history_tree):
    """Plans the best action given the belief state `b`."""
    q_values_log = []  # To track Q-values of each action-observation pair
    for i in range(550):  # Number of simulations
        s = sample_from_belief(b, P)
        SIMULATE(s, history_tree.root, d=12)

    # Find the best action node (child of the current root) based on Q-value
    best_child = max(history_tree.root.children, key=lambda node: node.q_value, default=None)
    print("#"*40)

    return best_child.action if best_child else sample_action(), history_tree


def sample_from_belief(b, P):
    """Samples a state from the belief distribution.
       The b is the mean of [x1,y1,theta, x2,y2,theta2,...] and P is the 3n*3n covariance matrix of these state values.
       
    Args:
    - b: A 3n x 1 mean vector, representing the belief (state mean).
    - P: A 3n x 3n covariance matrix representing the uncertainty in the state.

    Returns:
    - A 3n x 1 sampled state from the belief distribution.
    """
    # Sample from the multivariate normal distribution with mean `b` and covariance `P`
    sampled_state = np.random.multivariate_normal(b.flatten(), P)
    
    return sampled_state


# Helper Functions
def sample_action():
    """Generates a random action (v, w) for each robot.
       The linear velocity v and angular velocity w are sampled uniformly 
       between [-0.5, 0.5] for each robot.

    Args:
    - num_robots: Number of robots (default is 3).

    Returns:
    - A 2n x 1 array where each robot has its (v, w) action stacked.
    """
    # Continuous action settings
    # v_values = np.random.uniform(-0.5, 0.5, num_robots)  # Random linear velocities
    # w_values = np.random.uniform(-0.5, 0.5, num_robots)  # Random angular velocities

    # Define discretized ranges for v and w
    v_discrete = np.linspace(-0.5, 0.5, 6)
    w_discrete = np.linspace(-0.5, 0.5, 6)

    # Randomly sample linear and angular velocities from the discretized sets
    v_values = np.random.choice(v_discrete, num_robots)  # Random linear velocities from v_discrete
    w_values = np.random.choice(w_discrete, num_robots)  # Random angular velocities from w_discrete

    
    # Stack the velocities (v, w) for each robot into a single array
    actions = np.zeros((2 * num_robots, 1))
    actions[0::2] = v_values.reshape(-1, 1)  # Assign v values to even indices
    actions[1::2] = w_values.reshape(-1, 1)  # Assign w values to odd indices
    
    return actions


def R(env):
    """Computes the reward for a given state-action pair using specific target goals for each robot."""
    rewards = np.zeros(env.n_robots)
    
    # Loop through each robot and compute the reward based on its distance to its specific target goal
    for i, robot in enumerate(env.robots):
        goal_x = target_goals[2*i]   # x position of robot i's goal
        goal_y = target_goals[2*i + 1]  # y position of robot i's goal
        
        # Calculate the Euclidean distance from the robot's current position to its target goal
        dist_to_goal = np.linalg.norm([robot.x - goal_x, robot.y - goal_y])
        
        # Reward is negative distance, the closer the robot is to its goal, the higher the reward
        rewards[i] = -dist_to_goal  # Closer robots to their goal receive higher rewards

    return sum(rewards)



def plot_tree(history_tree, i =90):
    """Visualizes the history tree using Graphviz."""
    def add_nodes_edges(dot, node, parent_id=None, edge_label=None):
        """Recursively adds nodes and edges to the Graphviz object."""
        node_id = id(node)  # Use the memory address as a unique ID for the node
        label = f"Q: {node.q_value:.2f}\nVisits: {node.visit_count}"  # Node label with Q-value and visits
        
        if node.action is not None:
            label += f"\nAction: {node.action}"
        
        if node.observation is not None:
            label += f"\nObs: {node.observation}"
        
        dot.node(str(node_id), label)  # Add the node to the graph
        
        if parent_id is not None:
            dot.edge(str(parent_id), str(node_id), label=edge_label)  # Add an edge with optional label
        
        # Recursively add child nodes
        for child in node.children:
            add_nodes_edges(dot, child, parent_id=node_id, edge_label=f"A: {child.action}")
    filename=f'history_tree{i}'
    # Initialize the Graphviz object
    dot = Digraph(format="png")
    dot.attr(dpi="300")  # High resolution
    
    # Add the root node and recursively its children
    add_nodes_edges(dot, history_tree.root)
    
    # Render the graph to a file
    dot.render(filename, cleanup=True)
    print(f"Tree has been saved as {filename}{i}.png")

def compute_actions(Env, target_goals):
    """
    Compute the initial actions for all robots to move toward their respective goals.
    Each robot's control action is computed as (v, w) based on its current position and heading.
    
    Args:
    Env: The environment object containing robot states.
    target_goals: Array of goal positions for all robots [xg1, yg1, xg2, yg2, ...].
    
    Returns:
    actions: Array of shape (2*num_robots, 1) containing (v, w) for each robot.
    """
    num_robots = Env.n_robots
    actions = np.zeros((2 * num_robots, 1))
    states = Env.get_states()  # Get current states of the robots [x1, y1, theta1, ..., xn, yn, thetan]

    for i in range(num_robots):
        # Extract current state and goal for the ith robot
        x, y, theta = states[3 * i: 3 * i + 3]
        xg, yg = target_goals[2 * i: 2 * i + 2]
        

        # Define discretized ranges for v and w
        v_discrete = np.linspace(-0.5, 0.5, 6)
        w_discrete = np.linspace(-0.5, 0.5, 6)
        
        # Compute the error in position
        dx = xg - x
        dy = yg - y
        
        # Compute the desired heading angle and angular error
        desired_theta = np.arctan2(dy, dx)
        angle_error = desired_theta - theta
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))  # Normalize angle
        
        # Compute control actions (v, w) using a simple proportional controller
        v = max(-0.5, min(0.5,  0.02*np.sqrt(dx**2 + dy**2)))  # Proportional to distance to the goal
        w = max(-0.5, min(0.5, 0.07* angle_error   ))          # Proportional to angle error

        # Discretize (v, w) by finding the nearest values in the predefined ranges
        v_discretized = v_discrete[np.argmin(np.abs(v_discrete - v))]
        w_discretized = w_discrete[np.argmin(np.abs(w_discrete - w))]

        # Assign continuous actions to the actions array
        # actions[2 * i] = v
        # actions[2 * i + 1] = w

        # Assign discrete actions to the actions array
        actions[2 * i] = v_discretized
        actions[2 * i + 1] = w_discretized
    
    return actions

