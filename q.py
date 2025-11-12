import numpy as np
import random
import matplotlib.pyplot as plt
import math

# --- ENVIRONMENT SETUP ---
grid = 4
coordinates = [0, 1, 2, 3, 4] # Grid coordinates from 0 to 4
radius = 1.2

# Single disk movement options: (dx, dy)
MOVES = {
    0: (0, 0),  # Stay
    1: (0, 1),  # North
    2: (0, -1), # South
    3: (1, 0),  # East
    4: (-1, 0)  # West
}

# Generate 5x5 = 25 simultaneous actions: ((d1_dx, d1_dy), (d2_dx, d2_dy))
ACTIONS = {}
action_index = 0
for d1_move_idx in MOVES:
    for d2_move_idx in MOVES:
        # Action maps index to ((Disk 1 move vector), (Disk 2 move vector))
        ACTIONS[action_index] = (MOVES[d1_move_idx], MOVES[d2_move_idx])
        action_index += 1

DOT_DISTRIBUTION = [
    (1.73, 2.91), (1.33, 2.80), (1.25, 0.83), (2.59, 2.91),
    (1.64, 1.15), (2.01, 2.17), (1.60, 1.70), (0.47, 2.05),
    (1.81, 1.96), (1.64, 1.28), (2.01, 1.86), (2.94, 1.07),
    (1.26, 2.92), (1.56, 1.29), (2.03, 2.67), (1.89, 2.02),
    (1.76, 0.98), (1.74, 2.00), (1.69, 2.01), (2.45, 1.56),
    (0.39, 2.64), (3.27, 2.07), (3.27, 3.89), (2.89, 2.60),
    (0.60, 3.20)
]

def get_reward(state):
    (x1, y1), (x2, y2) = state
    covered_dots = set() # Use a set to avoid double counting

    for i, (dx, dy) in enumerate(DOT_DISTRIBUTION):
        # Disk 1 check
        distance_sq_1 = (dx - x1)**2 + (dy - y1)**2
        if distance_sq_1 <= radius**2:
            covered_dots.add(i)
            
        # Disk 2 check
        distance_sq_2 = (dx - x2)**2 + (dy - y2)**2
        if distance_sq_2 <= radius**2:
            covered_dots.add(i)
            
    return len(covered_dots)

def get_random_state():
    x1 = random.choice(coordinates)
    y1 = random.choice(coordinates)
    x2 = random.choice(coordinates)
    y2 = random.choice(coordinates)
    return ((int(x1), int(y1)), (int(x2), int(y2)))

# --- Q-LEARNING PARAMETERS ---
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate
num_episodes = 10000
MAX_STEPS_PER_EPISODE = 20 

# Generate all possible states (5^4 = 625 states)
states=[]
for x1 in coordinates:
    for y1 in coordinates:
        for x2 in coordinates:
            for y2 in coordinates:
                states.append( ((x1,y1),(x2,y2)) )

# Q-table initialization: (625 states x 25 actions)
Q = np.zeros((len(states), len(ACTIONS)))
state_to_index = {state: idx for idx, state in enumerate(states)}

def reset_environment():
    """Resets the environment by placing the disks at a random starting state."""
    return get_random_state()

def get_next_state(state, action):
    # Action = ((d1_dx, d1_dy), (d2_dx, d2_dy))
    (d1_move), (d2_move) = ACTIONS[action]
    (x1, y1), (x2, y2) = state
    grid_max = grid # grid is 4
    
    # Calculate new position for Disk 1 and ensure it's within bounds [0, 4]
    nx1 = np.clip(x1 + d1_move[0], 0, grid_max)
    ny1 = np.clip(y1 + d1_move[1], 0, grid_max)

    # Calculate new position for Disk 2 and ensure it's within bounds [0, 4]
    nx2 = np.clip(x2 + d2_move[0], 0, grid_max)
    ny2 = np.clip(y2 + d2_move[1], 0, grid_max)

    # Return the new state (positions must be integers as per coordinate definition)
    return ((int(nx1), int(ny1)), (int(nx2), int(ny2)))

def choose_action(state_index):
    if random.uniform(0, 1) < epsilon:
        return random.choice(list(ACTIONS.keys()))
    else:
        return np.argmax(Q[state_index]) # Exploit

def update_q_table(state_index, action, reward, next_state_index):
    old_q = Q[state_index, action]
    max_future_q = np.max(Q[next_state_index, :])
    
    # Q-Learning update rule
    new_q = (1 - alpha) * old_q + alpha * (reward + gamma * max_future_q)
    Q[state_index, action] = new_q

# --- THE EPISODIC TRAINING LOOP ---
print("Starting Q-Learning Training (25 Actions)...")

convergence_data = []
max_instantaneous_coverage_found = -1 
max_episode_return_found = -float('inf') 
CHECK_INTERVAL = 100

for episode in range(num_episodes):
    current_state = reset_environment()
    s = state_to_index[current_state]
    
    episode_return = 0
    
    for step in range(MAX_STEPS_PER_EPISODE):
        
        a = choose_action(s)
        
        new_state = get_next_state(current_state, a)
        s_prime = state_to_index[new_state]
        r = get_reward(new_state) # Step-wise reward
        
        episode_return += r # Accumulate step-wise reward
        
        update_q_table(s, a, r, s_prime)

        current_state = new_state
        s = s_prime

        if r > max_instantaneous_coverage_found:
            max_instantaneous_coverage_found = r

    # END OF EPISODE: Check the total accumulated reward (Return)
    if episode_return > max_episode_return_found:
        max_episode_return_found = episode_return

    if (episode + 1) % CHECK_INTERVAL == 0:
        convergence_data.append(max_episode_return_found)
        print(f"Episode {episode + 1}: Max Instantaneous Coverage = {max_instantaneous_coverage_found}, Max Accumulated Return = {max_episode_return_found}")

# --- FINDING THE OPTIMAL SOLUTION (Global Check) ---
print("\nTraining complete. Finding optimal positions...")

max_coverage = -1
optimal_state = None

# Iterate over all possible states to find the true global maximum coverage
for state in states:
    coverage = get_reward(state)
    
    if coverage > max_coverage:
        max_coverage = coverage
        optimal_state = state

print("\n--- RESULTS ---")
print(f"Total Dots Available: {len(DOT_DISTRIBUTION)}")
print(f"Maximum Dots Covered (Global Max): {max_coverage}")
print(f"Optimal Position for Disk 1 (x, y): {optimal_state[0]}")
print(f"Optimal Position for Disk 2 (x, y): {optimal_state[1]}")

# --- CONVERGENCE PLOT ---
print("\nPlotting convergence curve...")
x_axis = [i * CHECK_INTERVAL for i in range(1, len(convergence_data) + 1)]

plt.figure(figsize=(10, 6))
plt.plot(x_axis, convergence_data, marker='.', linestyle='-', color='b')
plt.title('Q-Learning Convergence Curve: Maximum Accumulated Episode Return (25 Actions)')
plt.xlabel('Training Episodes')
plt.ylabel('Max Accumulated Return Found')
plt.grid(True)
# Corrected axhline to use max_episode_return_found
plt.axhline(y=max_episode_return_found, color='r', linestyle='--', label=f'Final Max Return ({max_episode_return_found})') 
plt.legend()
plt.show()

# --- OPTIMAL DISK COVERAGE PLOT ---
def plot_disk_coverage(optimal_state, radius, dot_distribution, grid_size):
    (x1, y1), (x2, y2) = optimal_state

    fig, ax = plt.subplots(figsize=(8, 8))

    dot_xs = [d[0] for d in dot_distribution]
    dot_ys = [d[1] for d in dot_distribution]
    ax.scatter(dot_xs, dot_ys, color='gray', s=50, zorder=2, label='All Dots')

    # Plot Disk 1
    circle1 = plt.Circle((x1, y1), radius, color='red', alpha=0.3, label=f'Disk 1 Coverage')
    ax.add_patch(circle1)
    ax.scatter(x1, y1, color='red', marker='X', s=200, zorder=3, label=f'Disk 1 ({x1}, {y1})')

    # Plot Disk 2
    circle2 = plt.Circle((x2, y2), radius, color='blue', alpha=0.3, label=f'Disk 2 Coverage')
    ax.add_patch(circle2)
    ax.scatter(x2, y2, color='blue', marker='X', s=200, zorder=3, label=f'Disk 2 ({x2}, {y2})')

    # Identify and plot covered dots
    covered_dot_indices = set()
    for i, (dx, dy) in enumerate(dot_distribution):
        if (dx - x1)**2 + (dy - y1)**2 <= radius**2 or \
           (dx - x2)**2 + (dy - y2)**2 <= radius**2:
            covered_dot_indices.add(i)
    
    covered_xs = [dot_distribution[i][0] for i in covered_dot_indices]
    covered_ys = [dot_distribution[i][1] for i in covered_dot_indices]
    ax.scatter(covered_xs, covered_ys, color='green', s=100, zorder=4, label='Covered Dots')


    ax.set_xlim(-0.5, grid_size + 0.5)
    ax.set_ylim(-0.5, grid_size + 0.5)
    ax.set_xticks(np.arange(0, grid_size + 1, 1))
    ax.set_yticks(np.arange(0, grid_size + 1, 1))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title(f'Optimal Disk Coverage (Total Covered: {len(covered_dot_indices)})')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

# --- CALL THE NEW PLOTTING FUNCTION ---
print("\nPlotting optimal disk coverage...")
plot_disk_coverage(optimal_state, radius, DOT_DISTRIBUTION, grid)