from environment_discrete import DiscreteEnvironment
import numpy as np
from mcts import MCTS
from mcts_dpw import MCTS_DPW

import matplotlib.pyplot as plt
grid_size = 20
num_robots = 3
initial_positions = np.array([[18,15], [2,1], [12,1]])
goal_positions = np.array([[5, 3], [15, 15], [3,4]])

# Initialize environment
env = DiscreteEnvironment(num_robots, grid_size, initial_positions, goal_positions)
env_direct_to_goal = DiscreteEnvironment(num_robots, grid_size, initial_positions, goal_positions)
env_random = DiscreteEnvironment(num_robots, grid_size, initial_positions, goal_positions)
env_mcts = DiscreteEnvironment(num_robots, grid_size, initial_positions, goal_positions)
env_mcts_dpw = DiscreteEnvironment(num_robots, grid_size, initial_positions, goal_positions)
# mcts = MCTS(env_mcts, N={}, Q={}, d=10, m=70, c=5, U=env_mcts.calculate_U)
mcts = MCTS(env_mcts, N={}, Q={}, d=10, m=300, c=6, U=env_mcts.calculate_U)
mcts_dpw = MCTS_DPW(env_mcts_dpw, N={}, Q={}, d=10, m=300, c=6, U=env_mcts_dpw.calculate_U, state_action_pairs={}, available_actions={})
loss_arr_mcts_combined = []
loss_arr_mcts_dpw_combined = []
loss_arr_direct_combined = []
loss_random_combined = []
env.visualize_state(title='Initial Positions : MCTS')
print(env.robot_positions)
# Visualize initial state
for j in range(5):
    loss_arr_mcts = []
    loss_arr_direct = []
    loss_random = []
    loss_arr_dpw = []
    env = DiscreteEnvironment(num_robots, grid_size, initial_positions, goal_positions)
    env2 = DiscreteEnvironment(num_robots, grid_size, initial_positions, goal_positions)
    env_direct_to_goal = DiscreteEnvironment(num_robots, grid_size, initial_positions, goal_positions)
    env_random = DiscreteEnvironment(num_robots, grid_size, initial_positions, goal_positions)
    env_mcts = DiscreteEnvironment(num_robots, grid_size, initial_positions, goal_positions)
    env_mcts_dpw = DiscreteEnvironment(num_robots, grid_size, initial_positions, goal_positions)

    # mcts = MCTS(env_mcts, N={}, Q={}, d=10, m=70, c=5, U=env_mcts.calculate_U)
    mcts = MCTS(env_mcts, N={}, Q={}, d=10, m=200, c=6, U=env_mcts.calculate_U)
    mcts_dpw = MCTS_DPW(env_mcts_dpw, N={}, Q={}, d=10, m=200, c=6, U=env_mcts_dpw.calculate_U, state_action_pairs={}, available_actions={})

    for i in range(100):

        action = mcts(env.robot_positions)
        action_dpw = mcts_dpw(env2.robot_positions)
        action_direct_to_goal = env_direct_to_goal.direct_to_goal()
        env.step(action)
        env2.step(action_dpw)
        # env_random.random_rollout()
        env_direct_to_goal.step(action_direct_to_goal)
        loss_arr_mcts.append(env.calculate_reward())
        # loss_random.append(env_random.calculate_reward())
        loss_arr_dpw.append(env2.calculate_reward())
        loss_arr_direct.append(env_direct_to_goal.calculate_reward())
    loss_arr_direct_combined.append(loss_arr_direct)
    loss_arr_mcts_combined.append(loss_arr_mcts)
    # loss_random_combined.append(loss_random)
    loss_arr_mcts_dpw_combined.append(loss_arr_dpw)
print(env.robot_positions)
print(env_mcts_dpw.robot_positions)
# Convert lists to numpy arrays for easier manipulation
loss_arr_mcts_combined = np.array(loss_arr_mcts_combined)
loss_arr_direct_combined = np.array(loss_arr_direct_combined)
# loss_random_combined = np.array(loss_random_combined)
loss_arr_mcts_dpw_combined = np.array(loss_arr_mcts_dpw_combined)

# Calculate mean and 95% confidence interval
mean_loss_mcts = np.mean(loss_arr_mcts_combined, axis=0)
ci_loss_mcts = 1.96 * np.std(loss_arr_mcts_combined, axis=0) / np.sqrt(loss_arr_mcts_combined.shape[0])
mean_loss_mcts_dpw = np.mean(loss_arr_mcts_dpw_combined, axis=0)
ci_loss_mcts_dpw = 1.96 * np.std(loss_arr_mcts_dpw_combined, axis=0) / np.sqrt(loss_arr_mcts_dpw_combined.shape[0])

mean_loss_direct = np.mean(loss_arr_direct_combined, axis=0)
ci_loss_direct = 1.96 * np.std(loss_arr_direct_combined, axis=0) / np.sqrt(loss_arr_direct_combined.shape[0])
# mean_loss_random = np.mean(loss_random_combined, axis=0)
# ci_loss_random = 1.96 * np.std(loss_random_combined, axis=0) / np.sqrt(loss_random_combined.shape[0])

# Plotting with continuous spread of 95% CI
iterations = range(100)
plt.plot(iterations, mean_loss_mcts, label='MCTS')
plt.fill_between(iterations, mean_loss_mcts - ci_loss_mcts, mean_loss_mcts + ci_loss_mcts, alpha=0.2)
plt.plot(iterations, mean_loss_mcts_dpw, label='MCTS_DPW')
plt.fill_between(iterations, mean_loss_mcts_dpw - ci_loss_mcts_dpw, mean_loss_mcts_dpw + ci_loss_mcts_dpw, alpha=0.2)

plt.plot(iterations, mean_loss_direct, label='Direct')
plt.fill_between(iterations, mean_loss_direct - ci_loss_direct, mean_loss_direct + ci_loss_direct, alpha=0.2)
# plt.plot(iterations, mean_loss_random, label='Random Rollout')
# plt.fill_between(iterations, mean_loss_random - ci_loss_random, mean_loss_random + ci_loss_random, alpha=0.2)
plt.ylabel('Reward')
plt.xlabel('Iterations')
plt.legend()
plt.grid(True)
plt.title(f'Loss Comparison : 95% CI (depth={mcts_dpw.d}, m={mcts_dpw.m}, c={mcts_dpw.c})')

plt.show()

# print(env_direct_to_goal.robot_positions)
# plt.plot(loss_arr_mcts, label='MCTS')
# plt.plot(loss_arr_direct, label='Direct')
# plt.plot(loss_random, label='Random Rollout')
# plt.ylabel('Reward')
# plt.xlabel('Iterations')
# plt.legend()
# plt.grid(True)
# plt.title('Loss Comparison of Different Strategies')
# plt.show()
# env.visualize_state(title='Final Positions : MCTS')

# for k,v in mcts_dpw.available_actions.items():
#     print(k, len(v))

# images of snapshots of the robots 

# images of the loss being active

# run it for a few times and get the uncertaininty in it

# plot results using BAWS (lower bound)

# upper bound : direct

# make a action based grid showing the optimal action from each state