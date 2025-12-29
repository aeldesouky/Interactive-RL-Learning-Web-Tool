"""Custom Gymnasium Environments for RL Training.

This module provides custom environments including GridWorld and wrappers for additional environments.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SimpleGridWorld(gym.Env):
    """A simple grid world environment for reinforcement learning.
    
    The agent starts at position (0, 0) and must navigate to the goal at (grid_size-1, grid_size-1).
    The agent can move in 4 directions: up, down, left, right.
    
    Attributes:
        grid_size (int): Size of the square grid.
        agent_pos (tuple): Current position of the agent (row, col).
        goal_pos (tuple): Position of the goal state.
        max_steps (int): Maximum steps per episode.
        current_step (int): Current step counter.
    """
    
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 4}
    
    def __init__(self, grid_size=5, render_mode='rgb_array'):
        """Initialize the GridWorld environment.
        
        Args:
            grid_size (int): Size of the square grid (default: 5x5).
            render_mode (str): Rendering mode ('rgb_array' for visualization).
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Define action and observation spaces
        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        
        # Observations: (row, col) position in grid
        self.observation_space = spaces.Box(
            low=0, 
            high=grid_size-1, 
            shape=(2,), 
            dtype=np.int32
        )
        
        # Environment state
        self.agent_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)
        self.max_steps = grid_size * grid_size * 2
        self.current_step = 0
        
        # For rendering
        self.cell_size = 100
        
    def reset(self, seed=None, options=None, **kwargs):
        """Reset the environment to initial state.
        
        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional options (unused).
            
        Returns:
            tuple: (observation, info) where observation is the agent's position.
        """
        super().reset(seed=seed, options=options, **kwargs)
        
        # Reset agent to starting position
        self.agent_pos = (0, 0)
        self.current_step = 0
        
        observation = np.array(self.agent_pos, dtype=np.int32)
        info = {}
        
        return observation, info
    
    def step(self, action):
        """Take a step in the environment.
        
        Args:
            action (int): Action to take (0=up, 1=down, 2=left, 3=right).
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        
        # Get current position
        row, col = self.agent_pos
        
        # Move based on action
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(self.grid_size - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(self.grid_size - 1, col + 1)
        
        self.agent_pos = (row, col)
        
        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = 10.0  # Positive reward for reaching goal
            terminated = True
        else:
            reward = -0.1  # Small negative reward to encourage shorter paths
            terminated = False
        
        # Check if max steps reached
        truncated = self.current_step >= self.max_steps
        
        observation = np.array(self.agent_pos, dtype=np.int32)
        info = {}
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment.
        
        Returns:
            np.ndarray: RGB image array of the current state.
        """
        if self.render_mode != 'rgb_array':
            return None
        
        # Create RGB image
        img_size = self.grid_size * self.cell_size
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            pos = i * self.cell_size
            # Horizontal lines
            img[pos:pos+2, :] = [200, 200, 200]
            # Vertical lines
            img[:, pos:pos+2] = [200, 200, 200]
        
        # Draw goal (green)
        goal_row, goal_col = self.goal_pos
        start_row = goal_row * self.cell_size + 10
        end_row = start_row + self.cell_size - 20
        start_col = goal_col * self.cell_size + 10
        end_col = start_col + self.cell_size - 20
        img[start_row:end_row, start_col:end_col] = [0, 255, 0]
        
        # Draw agent (blue)
        if self.agent_pos is not None:
            agent_row, agent_col = self.agent_pos
            start_row = agent_row * self.cell_size + 20
            end_row = start_row + self.cell_size - 40
            start_col = agent_col * self.cell_size + 20
            end_col = start_col + self.cell_size - 40
            img[start_row:end_row, start_col:end_col] = [0, 0, 255]
        
        return img


class TreasureHuntWorld(gym.Env):
    """A treasure hunt grid world with obstacles and multiple treasures.
    
    The agent must navigate a grid to collect treasures while avoiding obstacles.
    The episode ends when all treasures are collected or max steps reached.
    
    Attributes:
        grid_size (int): Size of the square grid.
        agent_pos (tuple): Current position of the agent.
        treasures (set): Set of treasure positions remaining.
        obstacles (set): Set of obstacle positions.
    """
    
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 4}
    
    def __init__(self, grid_size=8, num_treasures=3, num_obstacles=5, render_mode='rgb_array'):
        """Initialize the TreasureHuntWorld environment.
        
        Args:
            grid_size (int): Size of the square grid.
            num_treasures (int): Number of treasures to place.
            num_obstacles (int): Number of obstacles to place.
            render_mode (str): Rendering mode.
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.num_treasures = num_treasures
        self.num_obstacles = num_obstacles
        self.render_mode = render_mode
        
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)
        
        self.agent_pos = (0, 0)
        self.treasures = set()
        self.initial_treasures = set()
        self.obstacles = set()
        self.max_steps = grid_size * grid_size * 3
        self.current_step = 0
        
        # Rendering
        self.cell_size = 80
        
        # Generate fixed layout
        self._generate_layout()
    
    def _generate_layout(self):
        rng = np.random.RandomState(42)
        all_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        all_positions.remove((0, 0))  # Remove start position
        
        rng.shuffle(all_positions)
        
        # Assign treasures and obstacles
        self.initial_treasures = set(all_positions[:self.num_treasures])
        self.obstacles = set(all_positions[self.num_treasures:self.num_treasures + self.num_obstacles])
    
    def reset(self, seed=None, options=None, **kwargs):
        """Reset the environment."""
        super().reset(seed=seed, options=options, **kwargs)
        
        self.agent_pos = (0, 0)
        self.treasures = self.initial_treasures.copy()
        self.current_step = 0
        
        observation = np.array(self.agent_pos, dtype=np.int32)
        info = {'treasures_remaining': len(self.treasures)}
        
        return observation, info
    
    def step(self, action):
        """Take a step in the environment."""
        self.current_step += 1
        
        # Get current position
        row, col = self.agent_pos
        
        # Attempt to move
        new_row, new_col = row, col
        if action == 0:  # up
            new_row = max(0, row - 1)
        elif action == 1:  # down
            new_row = min(self.grid_size - 1, row + 1)
        elif action == 2:  # left
            new_col = max(0, col - 1)
        elif action == 3:  # right
            new_col = min(self.grid_size - 1, col + 1)
        
        # Check if new position is valid (not an obstacle)
        new_pos = (new_row, new_col)
        if new_pos not in self.obstacles:
            self.agent_pos = new_pos
        
        # Calculate reward
        reward = -0.05  # Small time penalty
        
        # Check if collected treasure
        if self.agent_pos in self.treasures:
            self.treasures.remove(self.agent_pos)
            reward = 5.0  # Reward for collecting treasure
        
        # Check termination
        terminated = len(self.treasures) == 0
        if terminated:
            reward += 10.0  # Bonus for collecting all treasures
        
        truncated = self.current_step >= self.max_steps
        
        observation = np.array(self.agent_pos, dtype=np.int32)
        info = {'treasures_remaining': len(self.treasures)}
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode != 'rgb_array':
            return None
        
        # Create RGB image
        img_size = self.grid_size * self.cell_size
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            pos = i * self.cell_size
            img[pos:pos+1, :] = [220, 220, 220]
            img[:, pos:pos+1] = [220, 220, 220]
        
        # Draw obstacles (gray)
        for obs_row, obs_col in self.obstacles:
            start_row = obs_row * self.cell_size + 5
            end_row = start_row + self.cell_size - 10
            start_col = obs_col * self.cell_size + 5
            end_col = start_col + self.cell_size - 10
            img[start_row:end_row, start_col:end_col] = [100, 100, 100]
        
        # Draw treasures (yellow/gold)
        for tres_row, tres_col in self.treasures:
            start_row = tres_row * self.cell_size + 10
            end_row = start_row + self.cell_size - 20
            start_col = tres_col * self.cell_size + 10
            end_col = start_col + self.cell_size - 20
            img[start_row:end_row, start_col:end_col] = [255, 215, 0]
        
        # Draw agent (blue)
        if self.agent_pos is not None:
            agent_row, agent_col = self.agent_pos
            start_row = agent_row * self.cell_size + 15
            end_row = start_row + self.cell_size - 30
            start_col = agent_col * self.cell_size + 15
            end_col = start_col + self.cell_size - 30
            img[start_row:end_row, start_col:end_col] = [30, 144, 255]
        return img


class BreakoutLite(gym.Env):
    """A lightweight Breakout-style environment with discrete actions."""
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(self, width=12, height=14, paddle_width=3, render_mode='rgb_array'):
        super().__init__()
        self.width = width
        self.height = height
        self.paddle_width = paddle_width
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)  # 0 left, 1 stay, 2 right
        high_obs = np.array([width - 1, width - 1, height - 1, 1, 1, width * 2], dtype=np.int32)
        self.observation_space = spaces.Box(low=np.zeros(6, dtype=np.int32), high=high_obs, dtype=np.int32)
        self.max_steps = width * height * 2
        self.cell_size = 32
        self._reset_layout()

    def _reset_layout(self):
        self.paddle_x = self.width // 2
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_vx = 1
        self.ball_vy = -1
        self.bricks = {(r, c) for r in range(1, 3) for c in range(1, self.width - 1)}
        self.current_step = 0

    def reset(self, seed=None, options=None, **kwargs):
        super().reset(seed=seed, options=options, **kwargs)
        self._reset_layout()
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        if action == 0:
            self.paddle_x = max(1, self.paddle_x - 1)
        elif action == 2:
            self.paddle_x = min(self.width - 2, self.paddle_x + 1)
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        reward = -0.01
        terminated = False
        truncated = self.current_step >= self.max_steps
        if self.ball_x <= 0 or self.ball_x >= self.width - 1:
            self.ball_x = int(np.clip(self.ball_x, 1, self.width - 2))
            self.ball_vx *= -1
        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_vy *= -1
        paddle_row = self.height - 1
        paddle_cells = range(self.paddle_x - 1, self.paddle_x + self.paddle_width - 1)
        if self.ball_y >= paddle_row - 1 and self.ball_y <= paddle_row and self.ball_x in paddle_cells and self.ball_vy > 0:
            self.ball_vy *= -1
            reward += 0.1
        if (self.ball_y, self.ball_x) in self.bricks:
            self.bricks.remove((self.ball_y, self.ball_x))
            reward += 1.0
            self.ball_vy *= -1
        if not self.bricks:
            reward += 10.0
            terminated = True
        if self.ball_y > self.height:
            reward -= 1.0
            terminated = True
        return self._get_obs(), float(reward), terminated, truncated, {}

    def _get_obs(self):
        return np.array([
            self.paddle_x,
            self.ball_x,
            self.ball_y,
            self.ball_vx,
            self.ball_vy,
            len(self.bricks)
        ], dtype=np.int32)

    def render(self):
        if self.render_mode != 'rgb_array':
            return None
        img = np.ones((self.height * self.cell_size, self.width * self.cell_size, 3), dtype=np.uint8) * 255
        for r, c in self.bricks:
            sr, er = r * self.cell_size, (r + 1) * self.cell_size
            sc, ec = c * self.cell_size, (c + 1) * self.cell_size
            img[sr:er, sc:ec] = [255, 178, 102]
        paddle_row = self.height - 1
        for c in range(self.paddle_x - 1, self.paddle_x + self.paddle_width - 1):
            sr, er = paddle_row * self.cell_size, (paddle_row + 1) * self.cell_size
            sc, ec = c * self.cell_size, (c + 1) * self.cell_size
            if 0 <= c < self.width:
                img[sr:er, sc:ec] = [66, 135, 245]
        sr, er = self.ball_y * self.cell_size, (self.ball_y + 1) * self.cell_size
        sc, ec = self.ball_x * self.cell_size, (self.ball_x + 1) * self.cell_size
        sr = int(np.clip(sr, 0, img.shape[0] - 1))
        er = int(np.clip(er, sr + 1, img.shape[0]))
        sc = int(np.clip(sc, 0, img.shape[1] - 1))
        ec = int(np.clip(ec, sc + 1, img.shape[1]))
        img[sr:er, sc:ec] = [220, 53, 69]
        return img


class Gym4ReaLLite(gym.Env):
    """A simple moving-goal grid world inspired by real-world navigation tasks."""
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 6}

    def __init__(self, grid_size=6, render_mode='rgb_array'):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(4,), dtype=np.int32)
        self.max_steps = grid_size * grid_size * 2
        self.cell_size = 64
        self.rng = np.random.RandomState(123)
        self._reset_layout()

    def _reset_layout(self):
        self.agent_pos = (0, 0)
        self.goal_pos = (self.grid_size - 1, self.grid_size - 1)
        self.obstacles = {(1, 2), (2, 2), (3, 3)}
        self.current_step = 0

    def reset(self, seed=None, options=None, **kwargs):
        super().reset(seed=seed, options=options, **kwargs)
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self._reset_layout()
        return self._get_obs(), {'goal': self.goal_pos}

    def step(self, action):
        self.current_step += 1
        row, col = self.agent_pos
        new_row, new_col = row, col
        if action == 0:
            new_row = max(0, row - 1)
        elif action == 1:
            new_row = min(self.grid_size - 1, row + 1)
        elif action == 2:
            new_col = max(0, col - 1)
        elif action == 3:
            new_col = min(self.grid_size - 1, col + 1)
        new_pos = (new_row, new_col)
        if new_pos not in self.obstacles:
            self.agent_pos = new_pos
        reward = -0.05
        terminated = False
        truncated = self.current_step >= self.max_steps
        if self.agent_pos == self.goal_pos:
            reward = 8.0
            terminated = True
        else:
            self._move_goal()
        return self._get_obs(), float(reward), terminated, truncated, {'goal': self.goal_pos}

    def _move_goal(self):
        if self.current_step % 3 != 0:
            return
        gr, gc = self.goal_pos
        candidates = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = gr + dr, gc + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size and (nr, nc) not in self.obstacles:
                candidates.append((nr, nc))
        if candidates:
            self.goal_pos = candidates[self.rng.randint(len(candidates))]

    def _get_obs(self):
        return np.array([
            self.agent_pos[0],
            self.agent_pos[1],
            self.goal_pos[0],
            self.goal_pos[1]
        ], dtype=np.int32)

    def render(self):
        if self.render_mode != 'rgb_array':
            return None
        img_size = self.grid_size * self.cell_size
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        for i in range(self.grid_size + 1):
            pos = i * self.cell_size
            img[pos:pos+1, :] = [220, 220, 220]
            img[:, pos:pos+1] = [220, 220, 220]
        for obs_row, obs_col in self.obstacles:
            sr = obs_row * self.cell_size
            er = sr + self.cell_size
            sc = obs_col * self.cell_size
            ec = sc + self.cell_size
            img[sr:er, sc:ec] = [120, 120, 120]
        if self.goal_pos is not None:
            gr, gc = self.goal_pos
            sr = gr * self.cell_size + 8
            er = sr + self.cell_size - 16
            sc = gc * self.cell_size + 8
            ec = sc + self.cell_size - 16
            img[sr:er, sc:ec] = [40, 167, 69]
        if self.agent_pos is not None:
            ar, ac = self.agent_pos
            sr = ar * self.cell_size + 16
            er = sr + self.cell_size - 32
            sc = ac * self.cell_size + 16
            ec = sc + self.cell_size - 32
            img[sr:er, sc:ec] = [0, 123, 255]
        return img


def register_custom_environments():
    """Register custom environments with Gymnasium."""
    gym.register(id='SimpleGridWorld-v0', entry_point='custom_envs:SimpleGridWorld', max_episode_steps=50)
    gym.register(id='TreasureHuntWorld-v0', entry_point='custom_envs:TreasureHuntWorld', max_episode_steps=200)
    gym.register(id='BreakoutLite-v0', entry_point='custom_envs:BreakoutLite', max_episode_steps=300)
    gym.register(id='Gym4ReaLLite-v0', entry_point='custom_envs:Gym4ReaLLite', max_episode_steps=200)
    print("Custom environments registered: SimpleGridWorld-v0, TreasureHuntWorld-v0, BreakoutLite-v0, Gym4ReaLLite-v0")
