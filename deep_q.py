import numpy as np
import random
import matplotlib.pyplot as plt
import math

# --- ENVIRONMENT SETUP ---
grid = 4
coordinates = np.arange(0, 5) # [0, 1, 2, 3, 4]
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
    covered_dots = set() 
    # Logic is correct: set ensures no double counting
    for i, (dx, dy) in enumerate(DOT_DISTRIBUTION):
        if (dx - x1)**2 + (dy - y1)**2 <= radius**2:
            covered_dots.add(i)
        if (dx - x2)**2 + (dy - y2)**2 <= radius**2:
            covered_dots.add(i)
    return len(covered_dots)

def get_random_state():
    x1, y1 = random.choice(coordinates), random.choice(coordinates)
    x2, y2 = random.choice(coordinates), random.choice(coordinates)
    return ((int(x1), int(y1)), (int(x2), int(y2)))

# --- DQN PARAMETERS ---
alpha = 0.001 # Reduced learning rate for NN/Approximation
gamma = 0.9   # Discount factor
epsilon = 0.2 # Exploration rate
num_episodes = 15000 # Increased episodes for NN convergence
MAX_STEPS_PER_EPISODE = 25

# Generate all possible states (5^4 = 625 states)
states=[]
for x1 in coordinates:
    for y1 in coordinates:
        for x2 in coordinates:
            for y2 in coordinates:
                states.append( ((int(x1),int(y1)),(int(x2),int(y2))) )

state_to_index = {state: idx for idx, state in enumerate(states)}

# --- Q-NETWORK SIMULATION ---
# In a real DQN, this would be a large Keras/PyTorch model.
# Here, we use a simple NumPy array (Q_values) to store the learned Q-values,
# but the key difference is how we ACCESS and UPDATE it.
# We treat this array as the 'learned function' output, using the index
# mapping as the equivalent of the neural network's input/output mapping.
Q_values = np.zeros((len(states), len(ACTIONS)))


def predict_q_values(state_index):
    """Simulates the Q-Network forward pass: Input state, output Q-values for all actions."""
    # In a real DQN, this is NN.predict(state). Here, it's a table lookup.
    return Q_values[state_index, :]

def update_q_network(state_index, action, target_q):
    """Simulates training the Q-Network using one sample (s, a, target_q)."""
    # In a real DQN, this is where you perform a gradient descent step on the loss function.
    
    # Calculate the error (TD Error)
    predicted_q = Q_values[state_index, action]
    td_error = target_q - predicted_q
    
    # Update the 'Network weight' (the Q_value) via gradient descent approximation
    Q_values[state_index, action] += alpha * td_error
    
# --- ENVIRONMENT STEP FUNCTIONS ---

def reset_environment():
    return get_random_state()

def get_next_state(state, action):
    (d1_move), (d2_move) = ACTIONS[action]
    (x1, y1), (x2, y2) = state
    grid_max = grid
    
    nx1 = np.clip(x1 + d1_move[0], 0, grid_max)
    ny1 = np.clip(y1 + d1_move[1], 0, grid_max)
    nx2 = np.clip(x2 + d2_move[0], 0, grid_max)
    ny2 = np.clip(y2 + d2_move[1], 0, grid_max)

    return ((int(nx1), int(ny1)), (int(nx2), int(ny2)))

def choose_action(state_index):
    # Epsilon-Greedy policy uses the current predicted Q-values
    if random.uniform(0, 1) < epsilon:
        return random.choice(list(ACTIONS.keys()))
    else:
        q_predictions = predict_q_values(state_index)
        return np.argmax(q_predictions) # Exploit using the NN's output

# --- THE DQN TRAINING LOOP ---
print("Starting Deep Q-Learning Training (Simulated NN)...")

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
        r = get_reward(new_state)
        
        episode_return += r
        
        # --- DQN Update Step ---
        # 1. Predict Q-values for the next state (s')
        q_prime_predictions = predict_q_values(s_prime)
        max_future_q = np.max(q_prime_predictions)
        
        # 2. Calculate the Target (Y)
        target_q = r + gamma * max_future_q
        
        # 3. Update the Q-Network (simulated gradient step)
        update_q_network(s, a, target_q)
        # -------------------------

        current_state = new_state
        s = s_prime

        if r > max_instantaneous_coverage_found:
            max_instantaneous_coverage_found = r

    if episode_return > max_episode_return_found:
        max_episode_return_found = episode_return

    if (episode + 1) % CHECK_INTERVAL == 0:
        convergence_data.append(max_episode_return_found)
        print(f"Episode {episode + 1}: Max Instantaneous Coverage = {max_instantaneous_coverage_found}, Max Accumulated Return = {max_episode_return_found}")

# --- FINDING THE OPTIMAL SOLUTION (Global Check) ---
print("\nTraining complete. Finding optimal positions...")

max_coverage = -1
optimal_state = None

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
plt.title('DQN Convergence Curve: Maximum Accumulated Episode Return (Simulated)')
plt.xlabel('Training Episodes')
plt.ylabel('Max Accumulated Return Found')
plt.grid(True)
plt.axhline(y=max_episode_return_found, color='r', linestyle='--', label=f'Final Max Return ({max_episode_return_found})') 
plt.legend()
plt.show()

# --- OPTIMAL DISK COVERAGE PLOT (Visualization) ---
def plot_disk_coverage(optimal_state, radius, dot_distribution, grid_size):
    # ... (Plotting code remains the same as previous response) ...
    (x1, y1), (x2, y2) = optimal_state

    fig, ax = plt.subplots(figsize=(8, 8))

    dot_xs = [d[0] for d in dot_distribution]
    dot_ys = [d[1] for d in dot_distribution]
    ax.scatter(dot_xs, dot_ys, color='gray', s=50, zorder=2, label='All Dots')

    circle1 = plt.Circle((x1, y1), radius, color='red', alpha=0.3, label=f'Disk 1 Coverage')
    ax.add_patch(circle1)
    ax.scatter(x1, y1, color='red', marker='X', s=200, zorder=3, label=f'Disk 1 ({x1}, {y1})')

    circle2 = plt.Circle((x2, y2), radius, color='blue', alpha=0.3, label=f'Disk 2 Coverage')
    ax.add_patch(circle2)
    ax.scatter(x2, y2, color='blue', marker='X', s=200, zorder=3, label=f'Disk 2 ({x2}, {y2})')

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

print("\nPlotting optimal disk coverage...")
plot_disk_coverage(optimal_state, radius, DOT_DISTRIBUTION, grid)