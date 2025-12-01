Knee X-ray Classification

- 5-class KL grade classifier with 5-fold stratified CV using ResNet backbones.
- Portable CLI: no hardcoded paths, supports pretrained weights, mixed precision, and normalized/augmented inputs.

Project layout
- `data_preparation.py` – CLI to build `CSVs/` splits (train/val folds + test) from `metadata.csv`.
- `main.py` – trains 5 folds via `trainer.py`.
- `dataset.py` – dataset loader with train/val transforms, normalization, and optional image root.
- `model.py` – ResNet backbone (18/34/50) with 5-class head; optional ImageNet pretrained weights.
- `evaluate.py` – loads best fold checkpoints and reports recall metrics; saves `recall_results.csv` and `recall_plot.png`.
- `.gitignore` – keeps bulky data/artifacts out of git (CSVs, checkpoints, plots, images).

Quick start
- Python 3.9+ recommended.
- Env & deps: `python -m venv .venv && source .venv/bin/activate` (or `.\.venv\Scripts\activate` on Windows), then `pip install -r requirements.txt`.
- Data expectation: `metadata.csv` with columns `Name,Path,KL` where `Path` is absolute or relative to `--img_root`.

Data preparation
- Run from project root: `python data_preparation.py --data_root /path/to/dataset` (auto-writes CSVs to `/path/to/dataset/CSVs`).
- To customize CSV output: `python data_preparation.py --data_root /path/to/dataset --csv_dir /some/where/CSVs`.
- Regenerate CSVs on each machine so `Path` entries match that filesystem.

Training
- Example:  
  `python main.py --csv_dir /data/knee/CSVs --out_dir /data/knee/session --img_root /data/knee/images --backbone resnet18 --pretrained --batch_size 32 --learning_rate 1e-3 --epochs 10 --num_workers 4 --pin_memory --use_amp`
- Seeds are set (`--seed`) for reproducibility. Dataloaders honor `--num_workers`/`--pin_memory`.
- Uses normalization (ImageNet mean/std) and light augments for training (random resized crop, flip, jitter).
- Saves best checkpoint per fold to `out_dir/best_model_fold_{fold}.pth` and metric plots.

Evaluation
- Use the same paths used for training:  
  `python evaluate.py --csv_dir /data/knee/CSVs --out_dir /data/knee/session --img_root /data/knee/images --backbone resnet18`
- Writes `recall_results.csv` and `recall_plot.png` to the CWD.
- Evaluation uses the same resizing/normalization as validation.

Args reference (`args.py`)
- `--backbone {resnet18,resnet34,resnet50}`; `--pretrained` to load ImageNet weights.
- `--csv_dir PATH` CSV folds; `--img_root PATH` to resolve relative image paths.
- `--batch_size`, `--learning_rate`, `--epochs`, `--num_workers`, `--pin_memory`, `--use_amp`, `--seed`.
- `--device {auto,cuda,cpu}` (auto picks CUDA if available).
- `--out_dir PATH` for checkpoints/plots; `--save_best` flag available though best is saved by default.

Server/GitHub workflow
- Keep data/checkpoints out of git (see `.gitignore`).
- Push code to GitHub, then on the server: `git clone https://github.com/1avishek/Knee-Classification.git`, create a venv, `pip install -r requirements.txt`.
- Place dataset on the server, rerun `data_preparation.py` there with the server paths, then run training/eval commands with server-specific `--csv_dir`, `--out_dir`, and `--img_root`.

Troubleshooting
- File-not-found: ensure CSV `Path` entries exist on the machine; use `--img_root` or regenerate CSVs.
- CUDA/memory: lower `--batch_size`, disable `--use_amp`, or reduce `--num_workers`.
- Pretrained weights download: requires network the first time; omit `--pretrained` if offline.
