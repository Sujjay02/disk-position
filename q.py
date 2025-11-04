import numpy as np
import random
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
epsilon = 0.2  # Exploration rate (global variable)
num_episodes = 10000

# Generate all possible states (625 states)
states=[]
for x1 in coordinates:
    for y1 in coordinates:
        for x2 in coordinates:
            for y2 in coordinates:
                states.append( ((x1,y1),(x2,y2)) )

Q = np.zeros((len(states), len(ACTIONS)))
state_to_index = {state: idx for idx, state in enumerate(states)}


def get_next_state(state, action):
    (disk, direction) = ACTIONS[action]
    (x1, y1), (x2, y2) = state
    
    # Create mutable copies for the new position
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
    # Epsilon is a global variable, so no need to pass it
    if random.uniform(0, 1) < epsilon:
        return random.choice(list(ACTIONS.keys()))
    else:
        return np.argmax(Q[state_index]) # Exploit

def update_q_table(state_index, action, reward, next_state_index):
    # Use the passed arguments, not local variables s and a from the loop
    old_q = Q[state_index, action]
    
    # Find the maximum Q-value for the new state (s_prime)
    max_future_q = np.max(Q[next_state_index, :])
    
    # The new Q-value: (1-alpha)*old_q + alpha * [reward + gamma*max_future_q]
    new_q = (1 - alpha) * old_q + alpha * (reward + gamma * max_future_q)
    
    # Update the table
    Q[state_index, action] = new_q

# --- THE TRAINING LOOP ---
print("Starting Q-Learning Training...")

for episode in range(num_episodes):
    # 1. Get initial state
    current_state = get_random_state()
    s = state_to_index[current_state]
    
    # 2. Choose action (passing only one argument now)
    a = choose_action(s)
    
    # 3. Take action to get new state (s_prime) (FIXED LINE)
    new_state = get_next_state(current_state, a)
    s_prime = state_to_index[new_state] # FIXED VARIABLE
    
    # 4. Get Reward
    r = get_reward(new_state) # Reward based on the new state
    
    # 5. Update Q-Table (LEARN!)
    update_q_table(s, a, r, s_prime)

    if (episode + 1) % 1000 == 0:
        print(f"Completed Episode {episode + 1}")

# --- FINDING THE OPTIMAL SOLUTION ---
print("\nTraining complete. Finding optimal positions...")

max_coverage = -1
optimal_state = None

# Iterate over all possible states (FIXED VARIABLE)
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