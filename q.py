import numpy as np
import random
import matplotlib.pyplot as plt
import math

# --- ENVIRONMENT SETUP ---
grid = 4
coordinates = [0, 1, 2, 3, 4]
radius = 1.2
ACTIONS = {
    0: ('disk1', 'N'), 1: ('disk1', 'S'), 2: ('disk1', 'E'), 3: ('disk1', 'W'),
    4: ('disk2', 'N'), 5: ('disk2', 'S'), 6: ('disk2', 'E'), 7: ('disk2', 'W')
}

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
    covered_dots = set()

    for i, (dx, dy) in enumerate(DOT_DISTRIBUTION):
        # Disk 1 check
        distance_sq_1 = (dx - x1)**2 + (dy - y1)**2
        if distance_sq_1 <= radius**2:
            covered_dots.add(i)
            
        # Disk 2 check
        distance_sq_2 = (dx - x2)**2 + (dy - y2)**2
        if distance_sq_2 <= radius**2:
            covered_dots.add(i)
            
    # The reward is the number of unique dots covered
    return len(covered_dots)

def get_random_state():
    x1 = random.choice(coordinates)
    y1 = random.choice(coordinates)
    x2 = random.choice(coordinates)
    y2 = random.choice(coordinates)
    return ((x1, y1), (x2, y2))


# --- Q-LEARNING PARAMETERS ---
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate
num_episodes = 10000
MAX_STEPS_PER_EPISODE = 20 # Define the length of an episode

# Generate all possible states (625 states)
states=[]
for x1 in coordinates:
    for y1 in coordinates:
        for x2 in coordinates:
            for y2 in coordinates:
                states.append( ((x1,y1),(x2,y2)) )

Q = np.zeros((len(states), len(ACTIONS)))
state_to_index = {state: idx for idx, state in enumerate(states)}

def reset_environment():
    """Resets the environment by placing the disks at a random starting state."""
    return get_random_state()

def get_next_state(state, action):
    (disk, direction) = ACTIONS[action]
    (x1, y1), (x2, y2) = state
    
    nx1, ny1, nx2, ny2 = x1, y1, x2, y2

    if disk == 'disk1':
        if direction == 'N':
            ny1 = min(grid, y1 + 1)
        elif direction == 'S':
            ny1 = max(0, y1 - 1)
        elif direction == 'E':
            nx1 = min(grid, x1 + 1)
        elif direction == 'W':
            nx1 = max(0, x1 - 1)
        return ((nx1, ny1), (x2, y2))
    else:  # disk2
        if direction == 'N':
            ny2 = min(grid, y2 + 1)
        elif direction == 'S':
            ny2 = max(0, y2 - 1)
        elif direction == 'E':
            nx2 = min(grid, x2 + 1)
        elif direction == 'W':
            nx2 = max(0, x2 - 1)
        return ((x1, y1), (nx2, ny2))

def choose_action(state_index):
    if random.uniform(0, 1) < epsilon:
        return random.choice(list(ACTIONS.keys()))
    else:
        return np.argmax(Q[state_index]) # Exploit

def update_q_table(state_index, action, reward, next_state_index):
    old_q = Q[state_index, action]
    max_future_q = np.max(Q[next_state_index, :])
    new_q = (1 - alpha) * old_q + alpha * (reward + gamma * max_future_q)
    Q[state_index, action] = new_q

# --- THE TYPICAL EPISODIC TRAINING LOOP ---
print("Starting Q-Learning Training (Episodic Format)...")

# Convergence Tracking Variables
convergence_data = []
max_instantaneous_coverage_found = -1 # Tracks max step reward found
max_episode_return_found = -float('inf') # Tracks max accumulated reward found
CHECK_INTERVAL = 100

for episode in range(num_episodes):
    # 1. Start of Episode: Reset to a random initial state and reset the episode return
    current_state = reset_environment()
    s = state_to_index[current_state]
    
    episode_return = 0 # Initialize the total reward for this specific episode
    
    for step in range(MAX_STEPS_PER_EPISODE):
        
        # 2. Choose action (A)
        a = choose_action(s)
        
        # 3. Take action, observe reward (R) and new state (S')
        new_state = get_next_state(current_state, a)
        s_prime = state_to_index[new_state]
        r = get_reward(new_state) # Step-wise reward
        
        # Accumulate the step-wise reward to get the episode return
        episode_return += r
        
        # 4. Update Q-Table (LEARN!)
        update_q_table(s, a, r, s_prime)

        # 5. Transition to the new state
        current_state = new_state
        s = s_prime

        # Tracking instantaneous max
        if r > max_instantaneous_coverage_found:
            max_instantaneous_coverage_found = r

    # END OF EPISODE: Check the total accumulated reward (Return)
    if episode_return > max_episode_return_found:
        max_episode_return_found = episode_return

    # Convergence Tracking (at the end of each episode)
    if (episode + 1) % CHECK_INTERVAL == 0:
        # We track the highest *accumulated return* for the convergence plot
        convergence_data.append(max_episode_return_found)
        print(f"Episode {episode + 1}: Max Instantaneous Coverage = {max_instantaneous_coverage_found}, Max Accumulated Return = {max_episode_return_found}")

# --- FINDING THE OPTIMAL SOLUTION ---
print("\nTraining complete. Finding optimal positions...")

max_coverage = -1
optimal_state = None

# Iterate over all possible states (to ensure the true maximum is found)
for state in states:
    coverage = get_reward(state)
    
    if coverage > max_coverage:
        max_coverage = coverage
        optimal_state = state

print("\n--- RESULTS ---")
print(f"Total Dots Available: {len(DOT_DISTRIBUTION)}")
print(f"Maximum Dots Covered: {max_coverage}")
print(f"Optimal Position for Disk 1 (x, y): {optimal_state[0]}")
print(f"Optimal Position for Disk 2 (x, y): {optimal_state[1]}")


print("\nPlotting convergence curve...")

# Create the x-axis values (which are the episode numbers at each check point)
x_axis = [i * CHECK_INTERVAL for i in range(1, len(convergence_data) + 1)]

plt.figure(figsize=(10, 6))
plt.plot(x_axis, convergence_data, marker='.', linestyle='-', color='b')
plt.title('Q-Learning Convergence Curve: Maximum Accumulated Episode Return')
plt.xlabel('Training Episodes')
plt.ylabel('Max Accumulated Return Found')
plt.grid(True)
# FIXED: Using max_episode_return_found since the plot tracks the accumulated return
plt.axhline(y=max_episode_return_found, color='r', linestyle='--', label=f'Final Max Return ({max_episode_return_found})') 
plt.legend()
plt.show()

# --- NEW PLOTTING FUNCTION FOR DISK COVERAGE ---
def plot_disk_coverage(optimal_state, radius, dot_distribution, grid_size):
    (x1, y1), (x2, y2) = optimal_state

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot all dots
    dot_xs = [d[0] for d in dot_distribution]
    dot_ys = [d[1] for d in dot_distribution]
    ax.scatter(dot_xs, dot_ys, color='gray', s=50, zorder=2, label='All Dots')

    # Plot Disk 1
    circle1 = plt.Circle((x1, y1), radius, color='red', alpha=0.3, label=f'Disk 1 Coverage ({x1}, {y1})')
    ax.add_patch(circle1)
    ax.scatter(x1, y1, color='red', marker='X', s=200, zorder=3, label='Disk 1 Center')

    # Plot Disk 2
    circle2 = plt.Circle((x2, y2), radius, color='blue', alpha=0.3, label=f'Disk 2 Coverage ({x2}, {y2})')
    ax.add_patch(circle2)
    ax.scatter(x2, y2, color='blue', marker='X', s=200, zorder=3, label='Disk 2 Center')

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