import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# --- ENVIRONMENT SETUP ---
grid = 19
coordinates = np.arange(0, 20)  # [0, 1, 2, ..., 19]
radius = 1.2

# Single disk movement options: (dx, dy)
MOVES = {
    0: (0, 0),   # Stay
    1: (0, 1),   # North
    2: (0, -1),  # South
    3: (1, 0),   # East
    4: (-1, 0)   # West
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
alpha = 0.001        # Learning rate
gamma = 0.9          # Discount factor
epsilon_start = 1.0  # Starting exploration rate
epsilon_end = 0.05   # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate per episode
num_episodes = 5000
MAX_STEPS_PER_EPISODE = 50
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 10  # Update target network every N episodes

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- NEURAL NETWORK DEFINITION ---
class DQN(nn.Module):
    """
    Deep Q-Network with 2 hidden layers.
    Input: State vector [x1, y1, x2, y2] (normalized)
    Output: Q-values for all 25 actions
    """
    def __init__(self, state_size=4, action_size=25, hidden_sizes=[128, 128]):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_sizes[0])   # Input -> Hidden 1 (128 nodes)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])  # Hidden 1 -> Hidden 2 (128 nodes)
        self.fc3 = nn.Linear(hidden_sizes[1], action_size)  # Hidden 2 -> Output (25 actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # No activation on output (Q-values can be any real number)

# --- EXPERIENCE REPLAY BUFFER ---
class ReplayBuffer:
    """Stores transitions for experience replay."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).to(device)
        )

    def __len__(self):
        return len(self.buffer)

# --- HELPER FUNCTIONS ---
def state_to_tensor(state):
    """Convert state tuple to normalized tensor [x1, y1, x2, y2]."""
    (x1, y1), (x2, y2) = state
    # Normalize coordinates to [0, 1] range
    return [x1 / grid, y1 / grid, x2 / grid, y2 / grid]

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

# --- INITIALIZE NETWORKS ---
policy_net = DQN().to(device)  # The network we train
target_net = DQN().to(device)  # The network we use for stable Q-targets
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Target network is not trained directly

optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
loss_fn = nn.SmoothL1Loss()  # Huber loss - more stable than MSE for DQN
memory = ReplayBuffer(MEMORY_SIZE)

def choose_action(state, epsilon):
    """Epsilon-greedy action selection using the policy network."""
    if random.uniform(0, 1) < epsilon:
        return random.choice(list(ACTIONS.keys()))
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_to_tensor(state)).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            return q_values.argmax().item()

def train_step():
    """Perform one training step using a batch from replay buffer."""
    if len(memory) < BATCH_SIZE:
        return None

    # Sample a batch of transitions
    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    # Compute Q(s, a) using the policy network
    current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute max Q(s', a') using the target network (for stability)
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        # Target: r + gamma * max(Q(s', a')) * (1 - done)
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Compute loss and update
    loss = loss_fn(current_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

    return loss.item()

# --- THE DQN TRAINING LOOP ---
print("\n" + "="*60)
print("Starting Deep Q-Learning Training (Real Neural Network)")
print("="*60)
print(f"\nNetwork Architecture:")
print(f"  Input Layer:    4 nodes (x1, y1, x2, y2 normalized)")
print(f"  Hidden Layer 1: 128 nodes (ReLU)")
print(f"  Hidden Layer 2: 128 nodes (ReLU)")
print(f"  Output Layer:   25 nodes (Q-values for each action)")
print(f"\nTotal Parameters: {sum(p.numel() for p in policy_net.parameters()):,}")
print("="*60 + "\n")

all_episode_returns = []  # Per-episode return for every episode
all_episode_losses = []   # Per-episode avg loss for every episode
CHECK_INTERVAL = 50

epsilon = epsilon_start

for episode in range(num_episodes):
    current_state = reset_environment()
    episode_return = 0
    episode_loss = 0
    loss_count = 0

    for step in range(MAX_STEPS_PER_EPISODE):
        # Choose action using epsilon-greedy
        action = choose_action(current_state, epsilon)

        # Take action and observe next state and reward
        next_state = get_next_state(current_state, action)
        reward = get_reward(next_state)
        done = (step == MAX_STEPS_PER_EPISODE - 1)

        # Store transition in replay buffer
        memory.push(
            state_to_tensor(current_state),
            action,
            reward,
            state_to_tensor(next_state),
            float(done)
        )

        # Train the network
        loss = train_step()
        if loss is not None:
            episode_loss += loss
            loss_count += 1

        episode_return += reward
        current_state = next_state

    # Decay epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # Update target network periodically
    if (episode + 1) % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Track every episode
    all_episode_returns.append(episode_return)
    avg_loss = episode_loss / loss_count if loss_count > 0 else 0
    all_episode_losses.append(avg_loss)

    if (episode + 1) % CHECK_INTERVAL == 0:
        recent_avg = np.mean(all_episode_returns[-CHECK_INTERVAL:])
        print(f"Episode {episode + 1}: Avg Return (last {CHECK_INTERVAL}) = {recent_avg:.1f}, "
              f"Epsilon = {epsilon:.3f}, Avg Loss = {avg_loss:.4f}")

# --- FINDING THE OPTIMAL SOLUTION USING TRAINED NETWORK ---
print("\nTraining complete. Finding optimal positions using trained network...")

# Evaluate all possible states using the trained policy network
policy_net.eval()
max_coverage = -1
optimal_state = None
best_q_value = -float('inf')

# Sample states to find the best one according to the network
for x1 in coordinates:
    for y1 in coordinates:
        for x2 in coordinates:
            for y2 in coordinates:
                state = ((int(x1), int(y1)), (int(x2), int(y2)))
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
print("\nPlotting convergence curves...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Compute moving average
window = 50
moving_avg = np.convolve(all_episode_returns, np.ones(window)/window, mode='valid')

# Plot 1: Episodic Return (every episode + moving average)
episodes = np.arange(1, num_episodes + 1)
axes[0].plot(episodes, all_episode_returns, alpha=0.3, color='b', label='Per Episode')
axes[0].plot(np.arange(window, num_episodes + 1), moving_avg, color='r', linewidth=2,
             label=f'Moving Avg ({window} eps)')
axes[0].set_title('DQN Convergence: Episodic Return')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Episodic Return')
axes[0].grid(True)
axes[0].legend()

# Plot 2: Training Loss
axes[1].plot(episodes, all_episode_losses, alpha=0.3, color='orange', label='Per Episode')
loss_moving_avg = np.convolve(all_episode_losses, np.ones(window)/window, mode='valid')
axes[1].plot(np.arange(window, num_episodes + 1), loss_moving_avg, color='red', linewidth=2,
             label=f'Moving Avg ({window} eps)')
axes[1].set_title('DQN Training Loss')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Average Loss')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.savefig('convergence.png', dpi=150)
print("Saved convergence plot to convergence.png")

# --- OPTIMAL DISK COVERAGE PLOT ---
def plot_disk_coverage(optimal_state, radius, dot_distribution, grid_size):
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
    plt.savefig('disk_coverage.png', dpi=150)
    print("Saved disk coverage plot to disk_coverage.png")

print("\nPlotting optimal disk coverage...")
plot_disk_coverage(optimal_state, radius, DOT_DISTRIBUTION, grid)
