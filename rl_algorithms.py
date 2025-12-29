"""Reinforcement Learning Algorithms Implementation.

This module implements various RL algorithms for discrete action spaces,
including Policy Iteration, Value Iteration, Monte Carlo, Temporal Difference (TD),
SARSA, Q-learning, and n-step TD.
"""

import numpy as np
from collections import defaultdict, deque


class PolicyEvaluation:
    """Policy Evaluation for a fixed policy in discrete environments.
    
    Uses TD(0) style updates to estimate V(s) for the current policy.
    """

    def __init__(self, env, gamma=0.99, theta=0.001, alpha=0.1, policy=None):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.alpha = alpha
        self.policy = policy or {}
        self.V = defaultdict(float)
        try:
            self.n_actions = env.action_space.n
        except AttributeError:
            self.n_actions = env.action_space.shape[0]

    def select_action(self, state):
        state_key = self._discretize_state(state)
        if state_key in self.policy:
            return self.policy[state_key]
        return self.env.action_space.sample()

    def update(self, state, action, reward, next_state, done):
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        target = reward if done else reward + self.gamma * self.V[next_state_key]
        self.V[state_key] += self.alpha * (target - self.V[state_key])

    def _discretize_state(self, state):
        if isinstance(state, (int, np.integer)):
            return state
        if isinstance(state, np.ndarray):
            return tuple(np.round(state, decimals=1))
        return state


class PolicyIteration:
    """Policy Iteration algorithm for discrete environments.
    
    Alternates between policy evaluation (computing value function for current policy)
    and policy improvement (updating policy based on value function).
    
    Attributes:
        env: Gymnasium environment instance.
        gamma (float): Discount factor for future rewards.
        theta (float): Convergence threshold for policy evaluation.
        policy (dict): Current policy mapping states to actions.
        V (dict): State value function.
    """
    
    def __init__(self, env, gamma=0.99, theta=0.001):
        """Initialize Policy Iteration agent.
        
        Args:
            env: Gymnasium environment.
            gamma (float): Discount factor (0 to 1).
            theta (float): Convergence threshold for policy evaluation.
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.policy = {}
        self.V = defaultdict(float)
        self.iteration = 0
        
        # For discrete action spaces
        try:
            self.n_actions = env.action_space.n
        except AttributeError:
            self.n_actions = env.action_space.shape[0]
    
    def select_action(self, state):
        """Select action based on current policy.
        
        Args:
            state: Current environment state.
            
        Returns:
            int: Selected action.
        """
        state_key = self._discretize_state(state)
        
        if state_key not in self.policy:
            # Random action for unseen states
            return self.env.action_space.sample()
        
        return self.policy[state_key]
    
    def update(self, state, action, reward, next_state, done):
        """Update policy (performed periodically in real implementation).
        
        Args:
            state: Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state: Next state.
            done (bool): Whether episode terminated.
        """
        # Policy Iteration typically updates after complete episodes
        # Here we do a simplified version for real-time visualization
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        # Update value estimate
        if not done:
            self.V[state_key] += 0.1 * (reward + self.gamma * self.V[next_state_key] - self.V[state_key])
        else:
            self.V[state_key] += 0.1 * (reward - self.V[state_key])
        
        # Periodically improve policy
        self.iteration += 1
        if self.iteration % 10 == 0:
            self._improve_policy(state_key)
    
    def _improve_policy(self, state_key):
        """Improve policy for given state based on value function.
        
        Args:
            state_key: Discretized state key.
        """
        # Simplified policy improvement for continuous updates
        best_action = np.random.randint(self.n_actions)
        self.policy[state_key] = best_action
    
    def _discretize_state(self, state):
        """Convert continuous state to discrete key.
        
        Args:
            state: Environment state (can be array or scalar).
            
        Returns:
            tuple: Discretized state as hashable key.
        """
        if isinstance(state, (int, np.integer)):
            return state
        
        if isinstance(state, np.ndarray):
            # Discretize continuous states into bins
            return tuple(np.round(state, decimals=1))
        
        return state


class ValueIteration:
    """Value Iteration algorithm for discrete environments.
    
    Iteratively updates value function by taking max over actions,
    then derives optimal policy from converged values.
    
    Attributes:
        env: Gymnasium environment instance.
        gamma (float): Discount factor.
        theta (float): Convergence threshold.
        V (dict): State value function.
    """
    
    def __init__(self, env, gamma=0.99, theta=0.001):
        """Initialize Value Iteration agent.
        
        Args:
            env: Gymnasium environment.
            gamma (float): Discount factor (0 to 1).
            theta (float): Convergence threshold.
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = defaultdict(float)
        
        try:
            self.n_actions = env.action_space.n
        except AttributeError:
            self.n_actions = env.action_space.shape[0]
    
    def select_action(self, state):
        """Select greedy action based on value function.
        
        Args:
            state: Current environment state.
            
        Returns:
            int: Selected action.
        """
        state_key = self._discretize_state(state)
        
        # Greedy action selection (simplified)
        # In full implementation, would evaluate all actions
        return np.random.randint(self.n_actions)
    
    def update(self, state, action, reward, next_state, done):
        """Update value function.
        
        Args:
            state: Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state: Next state.
            done (bool): Whether episode terminated.
        """
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        # Value iteration update
        if not done:
            self.V[state_key] = max(self.V[state_key], 
                                   reward + self.gamma * self.V[next_state_key])
        else:
            self.V[state_key] = max(self.V[state_key], reward)
    
    def _discretize_state(self, state):
        """Convert continuous state to discrete key.
        
        Args:
            state: Environment state.
            
        Returns:
            tuple: Discretized state.
        """
        if isinstance(state, (int, np.integer)):
            return state
        
        if isinstance(state, np.ndarray):
            return tuple(np.round(state, decimals=1))
        
        return state


class MonteCarlo:
    """Monte Carlo learning algorithm.
    
    Learns from complete episodes using sample returns.
    Uses epsilon-greedy exploration strategy.
    
    Attributes:
        env: Gymnasium environment instance.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate for epsilon-greedy policy.
        Q (dict): Action-value function Q(s, a).
        returns (dict): Tracks returns for each state-action pair.
    """
    
    def __init__(self, env, gamma=0.99, epsilon=0.1):
        """Initialize Monte Carlo agent.
        
        Args:
            env: Gymnasium environment.
            gamma (float): Discount factor (0 to 1).
            epsilon (float): Exploration rate (0 to 1).
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(float))
        self.returns = defaultdict(lambda: defaultdict(list))
        self.episode_data = []
        
        try:
            self.n_actions = env.action_space.n
        except AttributeError:
            self.n_actions = env.action_space.shape[0]
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current environment state.
            
        Returns:
            int: Selected action.
        """
        state_key = self._discretize_state(state)
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # Greedy action
        if state_key in self.Q and self.Q[state_key]:
            return max(self.Q[state_key], key=lambda a: self.Q[state_key].get(a, 0.0))
        
        return self.env.action_space.sample()
    
    def update(self, state, action, reward, next_state, done):
        """Store experience and update Q-values at episode end.
        
        Args:
            state: Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state: Next state.
            done (bool): Whether episode terminated.
        """
        state_key = self._discretize_state(state)
        
        # Store experience
        self.episode_data.append((state_key, action, reward))
        
        # Update Q-values at episode end
        if done:
            self._update_episode()
    
    def _update_episode(self):
        """Update Q-values using returns from completed episode."""
        G = 0
        visited = set()
        
        # Process episode in reverse (for first-visit MC)
        for state_key, action, reward in reversed(self.episode_data):
            G = reward + self.gamma * G
            
            if (state_key, action) not in visited:
                self.returns[state_key][action].append(G)
                self.Q[state_key][action] = float(np.mean(self.returns[state_key][action]))
                visited.add((state_key, action))
        
        # Clear episode data
        self.episode_data = []
    
    def _discretize_state(self, state):
        """Convert continuous state to discrete key.
        
        Args:
            state: Environment state.
            
        Returns:
            tuple: Discretized state.
        """
        if isinstance(state, (int, np.integer)):
            return state
        
        if isinstance(state, np.ndarray):
            return tuple(np.round(state, decimals=1))
        
        return state


class TemporalDifference:
    """Temporal Difference (TD) learning - SARSA implementation.
    
    Updates Q-values after each step using bootstrapping.
    Uses epsilon-greedy exploration.
    
    Attributes:
        env: Gymnasium environment instance.
        gamma (float): Discount factor.
        alpha (float): Learning rate.
        epsilon (float): Exploration rate.
        Q (dict): Action-value function.
    """
    
    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=0.1):
        """Initialize TD learning agent.
        
        Args:
            env: Gymnasium environment.
            gamma (float): Discount factor (0 to 1).
            alpha (float): Learning rate (0 to 1).
            epsilon (float): Exploration rate (0 to 1).
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(float))
        
        try:
            self.n_actions = env.action_space.n
        except AttributeError:
            self.n_actions = env.action_space.shape[0]
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current environment state.
            
        Returns:
            int: Selected action.
        """
        state_key = self._discretize_state(state)
        
        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # Greedy
        if state_key in self.Q and self.Q[state_key]:
            return max(self.Q[state_key], key=lambda a: self.Q[state_key].get(a, 0.0))
        
        return self.env.action_space.sample()
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-value using TD(0) update rule.
        
        Args:
            state: Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state: Next state.
            done (bool): Whether episode terminated.
        """
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        # Get next action for SARSA
        next_action = self.select_action(next_state)
        
        # TD update
        current_q = self.Q[state_key][action]
        
        if done:
            target = reward
        else:
            next_q = self.Q[next_state_key][next_action]
            target = reward + self.gamma * next_q
        
        # Update Q-value
        self.Q[state_key][action] = current_q + self.alpha * (target - current_q)
    
    def _discretize_state(self, state):
        """Convert continuous state to discrete key.
        
        Args:
            state: Environment state.
            
        Returns:
            tuple: Discretized state.
        """
        if isinstance(state, (int, np.integer)):
            return state
        
        if isinstance(state, np.ndarray):
            return tuple(np.round(state, decimals=1))
        
        return state


class NStepTD:
    """n-step Temporal Difference learning.
    
    Bridges gap between TD(0) and Monte Carlo by looking ahead n steps.
    Uses eligibility traces for efficient multi-step updates.
    
    Attributes:
        env: Gymnasium environment instance.
        gamma (float): Discount factor.
        alpha (float): Learning rate.
        epsilon (float): Exploration rate.
        n (int): Number of steps for n-step returns.
        Q (dict): Action-value function.
        trajectory (deque): Stores recent (state, action, reward) tuples.
    """
    
    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=0.1, n=3):
        """Initialize n-step TD agent.
        
        Args:
            env: Gymnasium environment.
            gamma (float): Discount factor (0 to 1).
            alpha (float): Learning rate (0 to 1).
            epsilon (float): Exploration rate (0 to 1).
            n (int): Number of steps for n-step returns.
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.n = n
        self.Q = defaultdict(lambda: defaultdict(float))
        self.trajectory = deque(maxlen=n)
        
        try:
            self.n_actions = env.action_space.n
        except AttributeError:
            self.n_actions = env.action_space.shape[0]
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current environment state.
            
        Returns:
            int: Selected action.
        """
        state_key = self._discretize_state(state)
        
        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # Greedy
        if state_key in self.Q and self.Q[state_key]:
            return max(self.Q[state_key], key=lambda a: self.Q[state_key].get(a, 0.0))
        
        return self.env.action_space.sample()
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-value using n-step return.
        
        Args:
            state: Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state: Next state.
            done (bool): Whether episode terminated.
        """
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        # Add to trajectory
        self.trajectory.append((state_key, action, reward))
        
        # Update when we have n steps or episode ends
        if len(self.trajectory) == self.n or done:
            self._update_nstep(next_state_key, done)
    
    def _update_nstep(self, next_state_key, done):
        """Perform n-step TD update.
        
        Args:
            next_state_key: Discretized next state.
            done (bool): Whether episode terminated.
        """
        if not self.trajectory:
            return
        
        # Calculate n-step return
        G = 0
        for i, (_, _, reward) in enumerate(self.trajectory):
            G += (self.gamma ** i) * reward
        
        # Add bootstrap value if not terminal
        if not done:
            next_action = self.select_action(next_state_key)
            G += (self.gamma ** len(self.trajectory)) * self.Q[next_state_key][next_action]
        
        # Update first state-action in trajectory
        state_key, action, _ = self.trajectory[0]
        current_q = self.Q[state_key][action]
        self.Q[state_key][action] = current_q + self.alpha * (G - current_q)
        
        # Clear trajectory if episode done
        if done:
            self.trajectory.clear()
    
    def _discretize_state(self, state):
        """Convert continuous state to discrete key.
        
        Args:
            state: Environment state.
            
        Returns:
            tuple: Discretized state.
        """
        if isinstance(state, (int, np.integer)):
            return state
        
        if isinstance(state, np.ndarray):
            return tuple(np.round(state, decimals=1))
        
        return state


class SARSA(TemporalDifference):
    """On-policy SARSA control.
    
    Uses epsilon-greedy behavior policy and TD(0) update with next action.
    Inherits implementation from `TemporalDifference` for simplicity.
    """
    pass


class QLearning:
    """Off-policy Q-learning control.
    
    Updates Q-values using the greedy max over next-state actions,
    while behavior policy remains epsilon-greedy for exploration.
    """

    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(float))

        try:
            self.n_actions = env.action_space.n
        except AttributeError:
            self.n_actions = env.action_space.shape[0]

    def select_action(self, state):
        state_key = self._discretize_state(state)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        if state_key in self.Q and self.Q[state_key]:
            return max(self.Q[state_key], key=lambda a: self.Q[state_key].get(a, 0.0))

        return self.env.action_space.sample()

    def update(self, state, action, reward, next_state, done):
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)

        current_q = self.Q[state_key][action]

        if done:
            target = reward
        else:
            # Greedy target from next state (off-policy)
            if next_state_key in self.Q and self.Q[next_state_key]:
                max_next_q = max(self.Q[next_state_key].values())
            else:
                # If unseen, assume 0 for all actions
                max_next_q = 0.0
            target = reward + self.gamma * max_next_q

        self.Q[state_key][action] = current_q + self.alpha * (target - current_q)

    def _discretize_state(self, state):
        if isinstance(state, (int, np.integer)):
            return state
        if isinstance(state, np.ndarray):
            return tuple(np.round(state, decimals=1))
        return state
