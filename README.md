# InfluPose

PyTorch code for category-level object pose and size estimation from partial point clouds.

## Setup

Linux, Python 3.8, PyTorch 2.4.0, CUDA 11.8 and PyTorch3D 0.7.8.
A CUDA toolkit, C++ compiler and BF16-capable GPU are required.

```bash
conda env create -f environment.yml
conda activate influpose
bash scripts/install_geometry.sh
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
bash scripts/build_point_ops.sh
```

## Data

Prepare NOCS CAMERA and Real data with [SPD preprocessing](https://github.com/mentian/object-deformnet#datasets),
using the [GenPose directory layout](https://github.com/Jiyao06/GenPose#download-dataset-and-models).
Set `DATA_PATH` to the processed directory containing `CAMERA/`, `Real/`, `obj_models/`, and their split lists and labels. The default is `data/nocs/`.

Model checkpoints are distributed separately. Place them at:

```text
checkpoints/
├── stage1.pth
├── stage1_observability_audit.json
└── stage2.pth
```

## Training

```bash
# Stage 1
DATA_PATH=data/nocs GPU_IDS=0 BATCH_SIZE=32 OUTPUT_DIR=runs/stage1 \
  bash scripts/train_stage1.sh

# Teacher audit
DATA_PATH=data/nocs GPU_ID=0 CHECKPOINT=runs/stage1/epoch_036.pth \
  OUTPUT=runs/stage1/observability_audit.json \
  bash scripts/audit_stage1.sh

# Stage 2
bash scripts/train_stage2.sh \
  --data-path data/nocs \
  --stage1-checkpoint runs/stage1/epoch_036.pth \
  --stage1-audit runs/stage1/observability_audit.json \
  --output-dir runs/stage2 \
  --gpu-ids 0,1,2,3,4,5
```

## Evaluation and inference

```bash
DATA_PATH=data/nocs GPU_ID=0 OUTPUT_DIR=runs/eval_stage2 \
  bash scripts/eval_stage2_real275.sh

python scripts/infer.py --help
python -m pytest -q
```

## Project page

The static page is in `docs/`. Preview it locally with:

```bash
python -m http.server 8000 --directory docs
```
