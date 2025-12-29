# RL Training Visualizer

Flask + Socket.IO app to train tabular RL agents on discrete Gymnasium environments with live visualization and parameter tweaks.

## Quickstart

- Create a virtual env (recommended), then install deps: `pip install -r requirements.txt`.
- Launch the server: `python app.py` (serves on http://localhost:5000).
- In the UI: pick an environment, choose an algorithm, adjust params/episodes/delay, start training, and watch frames + rewards stream in.

## What’s Included

- Discrete Gymnasium tasks: CartPole-v1, MountainCar-v0, Blackjack-v1, FrozenLake-v1, CliffWalking-v0, Taxi-v3, plus custom SimpleGridWorld-v0, TreasureHuntWorld-v0, BreakoutLite-v0, and Gym4ReaLLite-v0 (auto-registered at startup).
- Eight tabular algorithms: Policy Evaluation, Policy Iteration, Value Iteration, Monte Carlo, Temporal Difference (TD), n-step TD, SARSA, Q-learning.
- Live controls: start/stop, adjustable animation delay, and runtime hyperparameter updates via Socket.IO.
- Visualization: rgb_array renders streamed to the browser with episode/step/reward overlays and basic charts.

## Usage Notes

- Algorithms assume **discrete action spaces**; pick compatible environments.
- Rendering uses `render_mode='rgb_array'`; ensure the selected env supports it.
- Box2D extras are intentionally omitted on Windows for build reliability; they are not required for the bundled tasks. If you need them, install separately with a working build toolchain.
- Socket.IO runs in threading mode for Python 3.13/Windows compatibility; no eventlet/gevent setup is needed.

## Custom Environments

- **SimpleGridWorld-v0**: 2D grid, reach bottom-right goal. Obs: row/col. Acts: up/down/left/right. Reward: +10 goal, -0.1/step.
- **TreasureHuntWorld-v0**: Grid with treasures and obstacles. Obs: row/col. Acts: up/down/left/right. Reward: +5 per treasure, +10 bonus when all collected, -0.05/step; fixed obstacle layout.
- **BreakoutLite-v0**: Discrete paddle control (left/stay/right) with falling ball and brick rows; rewards for brick hits, bonus when all cleared, penalty on miss.
- **Gym4ReaLLite-v0**: Moving-goal navigation with static obstacles; goal shifts periodically; step penalty and completion reward when reaching the goal.

## Running & Inspecting

- Start: `python app.py` → open the browser to the shown URL (default 5000).
- API helpers: `/api/environments` and `/api/algorithms` return the selectable options for the UI.
- Stop a run with the Stop button; parameters can be updated mid-run when training is active.

## Project Layout

```
app.py            # Flask + Socket.IO server and training loop
custom_envs.py    # SimpleGridWorld, TreasureHuntWorld, BreakoutLite, Gym4ReaLLite registrations
rl_algorithms.py  # Tabular RL agents (discrete-only, incl. policy evaluation)
templates/index.html
requirements.txt
README.md
```