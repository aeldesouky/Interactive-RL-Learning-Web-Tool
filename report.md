# RL Visualizer – Technical Report

## Scope
This report summarizes the implemented algorithms, custom and built-in environments, runtime parameter controls, and visualization/telemetry pipeline for the RL training visualizer.

## Algorithms (tabular, discrete action)
- Policy Evaluation – TD(0) value estimation for a fixed policy; parameters: `gamma`, `theta`, `alpha`; maintains `V`.
- Policy Iteration – iterative policy evaluation/improvement; value estimates stored in `V`; greedy action selection with periodic policy updates.
- Value Iteration – max-backup value updates; maintains `V`; greedy over inferred values.
- Monte Carlo (first-visit) – episodic returns; epsilon-greedy policy; maintains `Q` and per-(s,a) returns lists.
- Temporal Difference (TD/SARSA) – on-policy TD(0) with epsilon-greedy; maintains `Q`; bootstraps next action.
- n-step TD – on-policy multi-step returns; parameters: `n`, `gamma`, `alpha`, `epsilon`.
- SARSA – inherits TD logic; on-policy updates with epsilon-greedy.
- Q-learning – off-policy TD control; max over next-state actions; epsilon-greedy behavior policy.

## Environments
- Built-in (served via `/api/environments`): CartPole-v1, MountainCar-v0, Blackjack-v1, FrozenLake-v1, CliffWalking-v0, Taxi-v3.
- Custom (auto-registered):
  - SimpleGridWorld-v0 – grid navigation to bottom-right goal; obs: (row, col); acts: up/down/left/right; rewards: +10 goal, -0.1/step; max steps scaled to grid size.
  - TreasureHuntWorld-v0 – grid with treasures and obstacles; obs: (row, col); acts: up/down/left/right; rewards: +5 per treasure, +10 when all collected, -0.05/step; fixed obstacle layout; max steps scaled to grid size.
  - BreakoutLite-v0 – discrete paddle control (left/stay/right); ball/brick dynamics; rewards for brick hits, bonus on clear, penalty on miss; rgb grid render.
  - Gym4ReaLLite-v0 – moving-goal navigation with static obstacles; goal can drift; rewards for reaching goal, step penalty otherwise.
- Constraint: all bundled algorithms assume discrete action spaces; select compatible environments.

## Parameter Adjustment
- Catalog endpoint `/api/algorithms` returns per-algorithm parameter schemas (type, min, max, default, description) used to render UI controls.
- Algorithm-specific parameters: discount factor (gamma), learning rate (alpha), exploration rate (epsilon), convergence threshold (theta), n-step depth (n), and reward scaling (reward_scale) for real-time reward shaping experimentation.
- Start/stop: Socket.IO events `start_training` and `stop_training` create/stop a background training thread.
- Live hyperparameter updates: Socket.IO event `update_params` applies provided key/value pairs to the active agent instance during training; acknowledges with `params_updated` or reports errors.
- Client-side controls expose: algorithm parameters (per schema), number of episodes, animation delay, and environment selection.

## Visualization & Telemetry
- Rendering: environments created with `render_mode='rgb_array'`; frames converted to PNG via Pillow, base64-encoded, and sent to the frontend.
- Streaming events (Socket.IO):
  - `training_update`: includes episode, step, cumulative reward, action, value/Q proxy, done flag, and frame image.
  - `episode_complete`: episode total reward and step count.
  - `training_complete` / `training_error`: lifecycle notifications.
- Frontend (templates/index.html): displays live frame with overlay (episode/step/reward/status), episode history list, and charts for convergence/value snapshots/policy updates/decision traces; inference tab surfaces decision logs and behavior summaries.
- Performance knob: user-set animation delay throttles per-step emits; lowering delay speeds training but can stress rendering.

## Operational Notes
- Socket.IO runs in threading async mode for Windows/Python 3.13 compatibility; no eventlet/gevent required.
- Ensure chosen environments support `rgb_array` rendering; otherwise frames will be blank.
