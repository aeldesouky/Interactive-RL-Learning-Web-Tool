"""Flask application for visualizing reinforcement learning training.

This module provides a web interface for training and visualizing various
RL algorithms on Gymnasium environments.
"""

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import gymnasium as gym
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import threading
import time
from rl_algorithms import (
    PolicyEvaluation,
    PolicyIteration,
    ValueIteration,
    MonteCarlo,
    TemporalDifference,
    NStepTD,
    SARSA,
    QLearning
)
from custom_envs import register_custom_environments

# Register custom environments
register_custom_environments()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'rl-visualization-secret'
# Use threading async mode for Windows/Python 3.13 compatibility.
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Global training state
training_active = False
training_thread = None
current_agent = None


def render_frame(env):
    """Convert environment frame to base64 encoded image.
    
    Args:
        env: Gymnasium environment instance.
        
    Returns:
        str: Base64 encoded PNG image.
    """
    try:
        frame = env.render()
        if frame is None:
            return None

        # Normalize and validate frame shape before converting to image
        frame_arr = np.asarray(frame)

        # Gymnasium can occasionally emit lists/tuples; guard against malformed shapes
        if frame_arr.ndim < 2:
            return None

        # If values are float in [0,1], scale to [0,255]
        if np.issubdtype(frame_arr.dtype, np.floating) and frame_arr.max() <= 1.0:
            frame_arr = (frame_arr * 255).clip(0, 255).astype(np.uint8)
        else:
            frame_arr = frame_arr.astype(np.uint8, copy=False)

        # Convert RGB array to PIL Image
        img = Image.fromarray(frame_arr)
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error rendering frame: {e}")
        return None


def train_agent(env_name, algorithm, params, num_episodes, delay):
    """Training loop that emits updates via SocketIO.
    
    Args:
        env_name (str): Name of the Gymnasium environment.
        algorithm (str): Selected RL algorithm.
        params (dict): Algorithm parameters.
        num_episodes (int): Number of training episodes.
        delay (float): Delay between frames in seconds.
    """
    global training_active
    global current_agent
    
    try:
        # Create environment
        env = gym.make(env_name, render_mode='rgb_array')

        # Ensure environment has a discrete action space (algorithms assume discrete)
        try:
            from gymnasium.spaces import Discrete
            is_discrete = isinstance(env.action_space, Discrete)
        except Exception:
            is_discrete = hasattr(env.action_space, 'n')
        if not is_discrete:
            raise ValueError('Selected environment uses a continuous action space; please choose a discrete environment.')

        # Initialize algorithm
        algo_map = {
            'policy_evaluation': PolicyEvaluation,
            'policy_iteration': PolicyIteration,
            'value_iteration': ValueIteration,
            'monte_carlo': MonteCarlo,
            'temporal_difference': TemporalDifference,
            'nstep_td': NStepTD,
            'sarsa': SARSA,
            'q_learning': QLearning
        }
        reward_scale = float(params.pop('reward_scale', 1.0))
        
        agent = algo_map[algorithm](env, **params)
        # expose agent for live parameter updates
        current_agent = agent
        # store reward scaling for use within loop
        agent.reward_scale = reward_scale
        # expose agent for live parameter updates
        current_agent = agent
        
        # Training loop
        for episode in range(num_episodes):
            if not training_active:
                break
                
            state, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0.0
            step_count = 0
            
            while not done and not truncated and training_active:
                # Render current frame
                frame_data = render_frame(env)
                
                # Select action
                action = agent.select_action(state)
                
                # Take step
                next_state, reward, done, truncated, info = env.step(action)
                scaled_reward = float(reward) * getattr(agent, 'reward_scale', 1.0)
                
                # Update agent
                agent.update(state, action, scaled_reward, next_state, done or truncated)
                
                episode_reward += scaled_reward
                state = next_state
                step_count += 1
                
                # Extract telemetry for analytics
                try:
                    # Try to get discretized state key
                    if hasattr(agent, '_discretize_state'):
                        state_key = agent._discretize_state(state)
                    else:
                        state_key = str(state)
                    
                    # Try to get value estimate from agent's value function
                    if hasattr(agent, 'V'):
                        value_estimate = float(agent.V.get(state_key, 0.0))
                    else:
                        value_estimate = 0.0
                    
                    q_value = value_estimate  # Simplified: use V for Q-like estimate
                except Exception as e:
                    print(f"Warning: Failed to extract telemetry: {e}")
                    value_estimate = 0.0
                    q_value = 0.0
                
                # Emit update to frontend
                socketio.emit('training_update', {
                    'episode': episode + 1,
                    'step': step_count,
                    'reward': float(episode_reward),
                    'frame': frame_data,
                    'done': done or truncated,
                    'value': float(value_estimate),
                    'action': int(action),
                    'q_value': float(q_value)
                })
                
                # Delay for visualization
                time.sleep(delay)
            
            # Episode end update
            socketio.emit('episode_complete', {
                'episode': episode + 1,
                'total_reward': float(episode_reward),
                'steps': step_count
            })
        
        env.close()
        socketio.emit('training_complete', {'message': 'Training finished'})
        
    except Exception as e:
        print(f"Training error: {e}")
        socketio.emit('training_error', {'error': str(e)})
    
    finally:
        training_active = False
        # clear current agent reference
        current_agent = None


@socketio.on('update_params')
def handle_update_params(data):
    """Update agent parameters live during training.
    
    Args:
        data (dict): { 'params': { 'alpha': 0.2, 'epsilon': 0.05, ... } }
    """
    global current_agent, training_active
    params = data.get('params', {})

    if not training_active or current_agent is None:
        emit('training_error', {'error': 'No active training to update parameters'})
        return

    applied = {}
    for key, value in params.items():
        # Allow updating common hyperparameters if present on agent
        if hasattr(current_agent, key):
            try:
                setattr(current_agent, key, value)
                applied[key] = value
            except Exception as e:
                emit('training_error', {'error': f'Failed to set {key}: {e}'})

    if applied:
        emit('params_updated', {'applied': applied})


@app.route('/')
def index():
    """Render the main page.
    
    Returns:
        str: Rendered HTML template.
    """
    return render_template('index.html')


@app.route('/api/environments')
def get_environments():
    """Get list of available Gymnasium environments.
    
    Returns:
        JSON: List of environment names.
    """
    # Common environments suitable for RL visualization
    environments = [
        'CartPole-v1',
        'MountainCar-v0',
        'Blackjack-v1',
        'FrozenLake-v1',
        'CliffWalking-v0',
        'Taxi-v3',
        'SimpleGridWorld-v0',
        'TreasureHuntWorld-v0',
        'BreakoutLite-v0',
        'Gym4ReaLLite-v0'
    ]
    return jsonify(environments)


@app.route('/api/algorithms')
def get_algorithms():
    """Get list of available RL algorithms with their parameters.
    
    Returns:
        JSON: Dictionary of algorithms and their parameters.
    """
    algorithms = {
        'policy_evaluation': {
            'name': 'Policy Evaluation',
            'params': {
                'gamma': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.99,
                    'step': 0.01,
                    'description': 'Discount factor used when estimating state values under a fixed policy.'
                },
                'theta': {
                    'type': 'float',
                    'min': 0.0001,
                    'max': 0.1,
                    'default': 0.001,
                    'step': 0.0001,
                    'description': 'Convergence threshold for value stability when running evaluation.'
                },
                'alpha': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.1,
                    'step': 0.01,
                    'description': 'Learning rate for TD-style value updates.'
                },
                'reward_scale': {
                    'type': 'float',
                    'min': -5.0,
                    'max': 5.0,
                    'default': 1.0,
                    'step': 0.1,
                    'description': 'Multiply rewards on the fly to explore reward shaping.'
                }
            }
        },
        'policy_iteration': {
            'name': 'Policy Iteration',
            'params': {
                'gamma': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.99,
                    'step': 0.01,
                    'description': 'Discount factor - determines how much future rewards are valued. Higher values (closer to 1) make the agent prioritize long-term rewards.'
                },
                'theta': {
                    'type': 'float',
                    'min': 0.0001,
                    'max': 0.1,
                    'default': 0.001,
                    'step': 0.0001,
                    'description': 'Convergence threshold - smaller values lead to more accurate policy evaluation but slower convergence.'
                },
                'reward_scale': {
                    'type': 'float',
                    'min': -5.0,
                    'max': 5.0,
                    'default': 1.0,
                    'step': 0.1,
                    'description': 'Multiply rewards during training to explore reward shaping.'
                }
            }
        },
        'value_iteration': {
            'name': 'Value Iteration',
            'params': {
                'gamma': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.99,
                    'step': 0.01,
                    'description': 'Discount factor - balances immediate vs future rewards. Lower values make agent myopic, higher values encourage forward planning.'
                },
                'theta': {
                    'type': 'float',
                    'min': 0.0001,
                    'max': 0.1,
                    'default': 0.001,
                    'step': 0.0001,
                    'description': 'Convergence threshold - controls when to stop value iteration. Smaller values mean more precise value estimates.'
                },
                'reward_scale': {
                    'type': 'float',
                    'min': -5.0,
                    'max': 5.0,
                    'default': 1.0,
                    'step': 0.1,
                    'description': 'Multiply rewards during training to explore reward shaping.'
                }
            }
        },
        'monte_carlo': {
            'name': 'Monte Carlo (MC)',
            'params': {
                'gamma': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.99,
                    'step': 0.01,
                    'description': 'Discount factor - weighs future rewards. MC learns from complete episodes, so this affects return calculation.'
                },
                'epsilon': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.1,
                    'step': 0.01,
                    'description': 'Exploration rate - probability of taking random actions. Higher values increase exploration, lower values exploit learned policy.'
                },
                'reward_scale': {
                    'type': 'float',
                    'min': -5.0,
                    'max': 5.0,
                    'default': 1.0,
                    'step': 0.1,
                    'description': 'Reward multiplier for exploring reward shaping effects.'
                }
            }
        },
        'temporal_difference': {
            'name': 'Temporal Difference (TD)',
            'params': {
                'gamma': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.99,
                    'step': 0.01,
                    'description': 'Discount factor - determines time horizon for rewards. TD updates after each step using this for bootstrapping.'
                },
                'alpha': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.1,
                    'step': 0.01,
                    'description': 'Learning rate - controls how much new information overrides old. Higher values mean faster learning but more instability.'
                },
                'epsilon': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.1,
                    'step': 0.01,
                    'description': 'Exploration rate - balances exploration vs exploitation. Decays over time in epsilon-greedy strategies.'
                },
                'reward_scale': {
                    'type': 'float',
                    'min': -5.0,
                    'max': 5.0,
                    'default': 1.0,
                    'step': 0.1,
                    'description': 'Reward multiplier for exploring reward shaping effects.'
                }
            }
        },
        'nstep_td': {
            'name': 'n-step TD',
            'params': {
                'gamma': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.99,
                    'step': 0.01,
                    'description': 'Discount factor - used in n-step returns. Affects how future rewards compound over multiple steps.'
                },
                'alpha': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.1,
                    'step': 0.01,
                    'description': 'Learning rate - step size for value updates. Must be tuned carefully with n-step to avoid divergence.'
                },
                'epsilon': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.1,
                    'step': 0.01,
                    'description': 'Exploration rate - controls random action selection. Important for discovering optimal n-step paths.'
                },
                'n': {
                    'type': 'int',
                    'min': 1,
                    'max': 10,
                    'default': 3,
                    'step': 1,
                    'description': 'Number of steps - looks ahead n steps before updating. Larger n bridges gap between TD and MC, but increases variance.'
                },
                'reward_scale': {
                    'type': 'float',
                    'min': -5.0,
                    'max': 5.0,
                    'default': 1.0,
                    'step': 0.1,
                    'description': 'Reward multiplier for exploring reward shaping effects.'
                }
            }
        },
        'sarsa': {
            'name': 'SARSA',
            'params': {
                'gamma': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.99,
                    'step': 0.01,
                    'description': 'Discount factor - controls future reward weighting.'
                },
                'alpha': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.1,
                    'step': 0.01,
                    'description': 'Learning rate - step size for updates.'
                },
                'epsilon': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.1,
                    'step': 0.01,
                    'description': 'Exploration rate - probability of random action.'
                },
                'reward_scale': {
                    'type': 'float',
                    'min': -5.0,
                    'max': 5.0,
                    'default': 1.0,
                    'step': 0.1,
                    'description': 'Reward multiplier for exploring reward shaping effects.'
                }
            }
        },
        'q_learning': {
            'name': 'Q-learning',
            'params': {
                'gamma': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.99,
                    'step': 0.01,
                    'description': 'Discount factor - controls future reward weighting.'
                },
                'alpha': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.1,
                    'step': 0.01,
                    'description': 'Learning rate - step size for updates.'
                },
                'epsilon': {
                    'type': 'float',
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.1,
                    'step': 0.01,
                    'description': 'Exploration rate - probability of random action.'
                },
                'reward_scale': {
                    'type': 'float',
                    'min': -5.0,
                    'max': 5.0,
                    'default': 1.0,
                    'step': 0.1,
                    'description': 'Reward multiplier for exploring reward shaping effects.'
                }
            }
        }
    }
    return jsonify(algorithms)


@socketio.on('start_training')
def handle_start_training(data):
    """Start training with specified parameters.
    
    Args:
        data (dict): Training configuration including environment, algorithm, params, etc.
    """
    global training_active, training_thread
    
    if training_active:
        emit('training_error', {'error': 'Training already in progress'})
        return
    
    training_active = True
    
    env_name = data.get('environment')
    algorithm = data.get('algorithm')
    params = data.get('params', {})
    num_episodes = data.get('num_episodes', 10)
    delay = data.get('delay', 0.05)
    
    # Start training in separate thread
    training_thread = threading.Thread(
        target=train_agent,
        args=(env_name, algorithm, params, num_episodes, delay)
    )
    training_thread.start()
    
    emit('training_started', {'message': 'Training started'})


@socketio.on('stop_training')
def handle_stop_training():
    """Stop the current training session."""
    global training_active
    training_active = False
    emit('training_stopped', {'message': 'Training stopped'})


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
