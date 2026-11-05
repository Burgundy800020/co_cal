# Calibrating Offline Policies with Conformal Methods
CMU 10-703 project. Offline policy: TD3+BC on Minari MuJoCo (HalfCheetah, Hopper, Walker2d).

## Layout
- `src/cocal/` core modules (data, TD3+BC, evaluation, `conformal/`)
- `drivers/` run experiments, write artefacts to `runs/` (git-ignored)
- `analysis/` plotting/inference; read from `results/` (git-tracked), write to `results/figures/`
- `configs/` YAML hyperparameters; CLI overrides via `key=value`
- `notes/` research notes; `ref_papers/` papers (+ cached `.txt` in `ref_papers/txt/`)

## Workflow
```
conda env create -f environment.yml && conda activate cocal && pip install -e .
python drivers/train_offline.py configs/td3bc_hopper_medium.yaml train.steps=20000
cp -r runs/<name>/<timestamp> results/<experiment>     # promote a golden run
python analysis/plot_training.py results/<experiment>
```
