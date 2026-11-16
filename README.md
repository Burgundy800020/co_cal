# Failure-Efficient Online RL Fine-Tuning with Budgeted Expert Intervention
CMU 10-703 project (package `safe_f2o`). Online failures/terminations are the costly part of online RL
fine-tuning. We fine-tune an offline policy while a budgeted expert takes over when the policy looks
unreliable (e.g. low Q), as in ThriftyDAgger; fine-tuning combines online TD3 with BC on expert steps.
Main metric: episode return vs. number of online terminations (baselines: offline-to-online TD3, DAgger).
Full plan in `claude/reorganize.md`.

Offline pretraining: TD3+BC on Minari MuJoCo (HalfCheetah, Hopper, Walker2d).

## Layout
- `src/safe_f2o/` core modules (data, TD3+BC, evaluation, `checkpoint` (agent/run-dir loading),
  `online` (env-agnostic online loop with termination counting, pluggable reset), `intervention/`)
- `drivers/` run experiments, write artefacts to `runs/` (git-ignored)
- `analysis/` plotting/inference; read from `results/` (git-tracked), write to `results/figures/`
- `configs/` YAML hyperparameters; CLI overrides via `key=value`
- `notes/` research notes; `ref_papers/` papers (+ cached `.txt` in `ref_papers/txt/`)

## Workflow
```
conda env create -f environment.yml && conda activate safe_f2o && pip install -e .
python drivers/train_offline.py configs/td3bc_hopper_medium.yaml train.steps=20000
cp -r runs/<name>/<timestamp> results/<experiment>     # promote a golden run
python analysis/plot_training.py results/<experiment>
# online fine-tuning (unlimited budget) from an offline run -> expert phase 1 / O2O baseline
python drivers/train_online.py configs/online_td3_ant_medium.yaml [online.total_steps=...]
```
