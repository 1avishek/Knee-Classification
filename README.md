Knee X-ray Classification

- Trains a 5-class KL grade classifier on knee X-ray images using ResNet backbones and 5-fold stratified CV.
- Code is organized for local training, evaluation, and quick plotting of metrics.

Project layout
- `data_preparation.py` – builds `CSVs/` splits (train/val folds + test) from `metadata.csv`.
- `main.py` – trains 5 folds via `trainer.py`.
- `dataset.py` – dataset and preprocessing (OpenCV read → PIL → tensor).
- `model.py` – ResNet backbone with 5-class head.
- `evaluate.py` – loads best fold checkpoints and reports recall metrics; saves `recall_results.csv` and `recall_plot.png`.
- `requirements.txt` – core Python deps (install OpenCV separately).

Quick start (local)
- Python 3.9+ recommended.
- Create env and install deps: `python -m venv .venv && source .venv/bin/activate` (or `.\.venv\Scripts\activate` on Windows), then `pip install -r requirements.txt opencv-python`.
- Ensure `metadata.csv` exists with columns `Name,Path,KL` where `Path` points to each image.

Data preparation
- Update `data_preparation.py` to set `main_dir` to your dataset root on the target machine.
- Run `python data_preparation.py` to create `CSVs/test.csv` and `CSVs/fold_{0-4}_{train,val}.csv` plus distribution plots.
- If moving to another machine, regenerate the CSVs there so `Path` values match that filesystem.

Training
- Defaults in `args.py` point to Windows paths; override on the CLI for portability:
  - Example: `python main.py --csv_dir /home/user/knee_classification/CSVs --out_dir /home/user/knee_classification/session --backbone resnet18 --batch_size 32 --learning_rate 1e-3`
- Models for each fold are saved to `out_dir` as `best_model_fold_{fold}.pth`; training/validation plots are saved alongside.

Evaluation
- After training, run: `python evaluate.py --csv_dir /home/user/knee_classification/CSVs --out_dir /home/user/knee_classification/session`
- Produces `recall_results.csv` and `recall_plot.png` in the working directory.

Args reference (`args.py`)
- `--backbone {resnet18,resnet34,resnet50,small_cnn}` (small_cnn not implemented; defaults to resnet18/34/50).
- `--csv_dir PATH` location of generated CSV folds.
- `--batch_size {8,16,32,64,128}` default 32.
- `--learning_rate FLOAT` default 1e-3.
- `--out_dir PATH` where checkpoints/plots go.
- `--epochs INT` exposed but `trainer.py` currently uses a hardcoded 10; adjust there if needed.
- `--device` exposed but device auto-selects CUDA if available in `trainer.py`.

Porting to a server via GitHub
- Commit and push this repo to GitHub (avoid adding raw image data if large; consider `.gitignore` for datasets).
- On the server: `git clone https://github.com/<you>/<repo>.git` (or `git pull` to update).
- Install deps on the server env: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt opencv-python`.
- Place your dataset on the server and rerun `data_preparation.py` there with the correct `main_dir` so CSV `Path` entries are valid.
- Run training/eval commands with server-specific `--csv_dir` and `--out_dir` paths.

Troubleshooting
- ImportError for `cv2`: install `opencv-python`.
- File-not-found during loading: confirm CSV `Path` entries exist on the current machine; regenerate splits if paths differ.
- UnicodeEncodeError when printing: some scripts contain non-ASCII characters; replace them with plain text if your terminal cannot render them.
