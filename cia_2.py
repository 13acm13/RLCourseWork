import numpy as np
import random
from collections import defaultdict, deque
import matplotlib.pyplot as plt

class MazeEnvironment:
    def __init__(self, size=100, obstacle_count=100):
        self.size = size
        self.maze = np.zeros((size, size), dtype=int)
        self.start_pos = None
        self.goal_pos = None
        self.obstacle_count = obstacle_count
        self.generate_obstacles()
        self.set_start_goal()

    def generate_obstacles(self):
        count = 0
        while count < self.obstacle_count:
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if (x, y) != (0, 0) and (x, y) != (self.size - 1, self.size - 1) and self.maze[x, y] == 0:
                self.maze[x, y] = 1
                if not self.path_exists():
                    self.maze[x, y] = 0
                else:
                    count += 1

    def path_exists(self):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        start_pos = (0, 0)
        goal_pos = (self.size - 1, self.size - 1)
        visited = set()
        queue = deque([start_pos])

        while queue:
            x, y = queue.popleft()
            if (x, y) == goal_pos:
                return True
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.size and
                    0 <= new_y < self.size and
                    self.maze[new_x, new_y] == 0 and
                    (new_x, new_y) not in visited):
                    visited.add((new_x, new_y))
                    queue.append((new_x, new_y))
        return False

    def set_start_goal(self):
        free_spaces = [(i, j) for i in range(self.size) for j in range(self.size) if self.maze[i][j] == 0]
        self.start_pos = random.choice(free_spaces)
        free_spaces.remove(self.start_pos)
        self.goal_pos = random.choice(free_spaces)

        # Set the start and goal positions
        self.maze[self.start_pos] = -1
        self.maze[self.goal_pos] = -2

class ReinforcementLearningAgent:
    def __init__(self, environment, discount_factor=0.99, learning_rate=0.1, exploration_rate=0.1):
        self.environment = environment
        self.state_space = [(i, j) for i in range(environment.size) for j in range(environment.size) if environment.maze[i][j] != -1]
        self.action_space = ['up', 'down', 'left', 'right']
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.q_values = defaultdict(lambda: defaultdict(float))
        self.rewards = self._initialize_rewards()

    def _initialize_rewards(self):
        rewards = {}
        for state in self.state_space:
            if state == self.environment.goal_pos:
                rewards[state] = 100
            else:
                rewards[state] = -1
        return rewards

    def _get_next_state(self, current_state, action):
        i, j = current_state
        if action == 'up':
            i -= 1
        elif action == 'down':
            i += 1
        elif action == 'left':
            j -= 1
        elif action == 'right':
            j += 1

        if 0 <= i < self.environment.size and 0 <= j < self.environment.size and self.environment.maze[i][j] != -1:
            return (i, j)
        else:
            return current_state

    def select_action(self, current_state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(self.action_space)
        else:
            q_values = self.q_values[current_state]
            max_q_value = max(q_values.values(), default=0)
            best_actions = [action for action, q in q_values.items() if q == max_q_value]
            return random.choice(best_actions)

    def update_q_values(self, episodes=1000, max_steps_per_episode=100):
        for episode in range(episodes):
            current_state = self.environment.start_pos
            steps = 0

            while current_state != self.environment.goal_pos and steps < max_steps_per_episode:
                action = self.select_action(current_state)
                next_state = self._get_next_state(current_state, action)
                reward = self.rewards[next_state]

                max_future_q = max(self.q_values[next_state].values(), default=0)
                td_target = reward + self.discount_factor * max_future_q
                td_delta = td_target - self.q_values[current_state][action]
                self.q_values[current_state][action] += self.learning_rate * td_delta

                current_state = next_state
                steps += 1

            self.exploration_rate = max(0.01, self.exploration_rate * 0.995)

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{episodes} completed.")

            if current_state == self.environment.goal_pos:
                print(f"Goal reached in episode {episode + 1} after {steps} steps.")

def visualize_policy(agent):
    maze_copy = np.copy(agent.environment.maze)
    for state in agent.state_space:
        i, j = state
        best_action = agent.select_action(state)

        # Arrow direction based on best action
        dx, dy = (0, -0.5) if best_action == 'up' else (0, 0.5) if best_action == 'down' else (-0.5, 0) if best_action == 'left' else (0.5, 0)
        plt.arrow(j, i, dx, dy, head_width=0.2, head_length=0.2, fc='orange', ec='orange')

    plt.imshow(maze_copy, cmap='gray_r', origin='upper')
    plt.scatter(agent.environment.start_pos[1], agent.environment.start_pos[0], c='green', label='Start', s=100)
    plt.scatter(agent.environment.goal_pos[1], agent.environment.goal_pos[0], c='red', label='Goal', s=100)
    plt.title("Learned Policy Visualization")
    plt.legend()
    plt.grid(True)
    plt.show()

# Create and train the Reinforcement Learning agent
maze_env = MazeEnvironment()
rl_agent = ReinforcementLearningAgent(maze_env)
rl_agent.update_q_values(episodes=1000)

# Visualize the learned policy
visualize_policy(rl_agent)
