"""Entry point packages for UAV-Jammer-RL.

Training modules live under `Main.train`, for example:
- `python -m Main.train.train_iql`
- `python -m Main.train.train_vdn`
- `python -m Main.train.train_qmix`
- `python -m Main.train.train_qplex`
- `python -m Main.train.train_mappo`
- `python -m Main.train.train_qmix_value_expansion`
- `python -m Main.train.train_world_model`

Evaluation modules live under `Main.evaluate`, for example:
- `python -m Main.evaluate.evaluate_all_baselines --episodes 100 --steps 1000`
- `python -m Main.evaluate.run_heuristic --policy greedy_sensing`
- `python -m Main.evaluate.evaluate_mpdqn --mode mpdqn --weights <weights.pth>`
- `python -m Main.evaluate.evaluate_mappo --weights <mappo_weights.pth>`
"""
