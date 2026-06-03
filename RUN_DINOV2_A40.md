# DINOv2 Chair Training On 2xA40

This is the recommended path for the new Blender Objaverse chair dataset.

## 1. Prepare Environment

```bash
cd /data
bash /data/repositoryi/setup_a40_env.sh
source /data/venv/bin/activate
```

If the repository is not present yet, clone/update it manually before this step.

## 2. Download Dataset From Kaggle

Put `kaggle.json` at:

```bash
mkdir -p ~/.kaggle
chmod 700 ~/.kaggle
# place ~/.kaggle/kaggle.json here
chmod 600 ~/.kaggle/kaggle.json
```

Download:

```bash
mkdir -p /data/datasets/objaverse-chair-blender-dataset
kaggle datasets download \
  -d neixon/objaverse-chair-blender-dataset \
  -p /data/datasets/objaverse-chair-blender-dataset \
  --unzip
```

Expected:

```text
/data/datasets/objaverse-chair-blender-dataset/metadata/views.csv
/data/datasets/objaverse-chair-blender-dataset/renders/<uid>/view_000.png
/data/datasets/objaverse-chair-blender-dataset/masks/<uid>/view_000.png
/data/datasets/objaverse-chair-blender-dataset/depths/<uid>/view_000.exr
/data/datasets/objaverse-chair-blender-dataset/normals/<uid>/view_000.png
/data/datasets/objaverse-chair-blender-dataset/objects/<uid>/points.npz
```

## 3. Train Strong DINOv2 Triplane Model

Stable first run:

```bash
source /data/venv/bin/activate

torchrun --standalone --nproc_per_node=2 /data/repositoryi/train_chair_dinov2_triplane_cuda.py \
  --mode train \
  --dataset_root /data/datasets/objaverse-chair-blender-dataset \
  --work_dir /data/runs/chair_dinov2_triplane \
  --image_size 518 \
  --dinov2_model dinov2_vitl14_reg \
  --batch_size 1 \
  --grad_accum 8 \
  --queries_per_item 49152 \
  --val_queries 24576 \
  --surface_points 8192 \
  --plane_size 128 \
  --plane_channels 64 \
  --decoder_hidden 512 \
  --decoder_layers 5 \
  --latent_dim 1024 \
  --epochs 60 \
  --lr 8e-5 \
  --encoder_lr 2e-6 \
  --unfreeze_encoder_epoch 20 \
  --amp bf16 \
  --num_workers 12 \
  --require_cuda \
  --skip_install
```

If VRAM is too tight, change:

```bash
--dinov2_model dinov2_vitb14_reg
--latent_dim 768
--queries_per_item 32768
--plane_channels 48
```

If it is comfortably under memory and fast enough, try:

```bash
--queries_per_item 65536
--surface_points 16384
--plane_channels 80
```

## 4. Predict Mesh From One Image

```bash
source /data/venv/bin/activate

python /data/repositoryi/train_chair_dinov2_triplane_cuda.py \
  --mode predict \
  --checkpoint /data/runs/chair_dinov2_triplane/best.pt \
  --image /data/my_chair.png \
  --output_dir /data/runs/chair_dinov2_test \
  --grid_resolution 160 \
  --level 0.025 \
  --skip_install
```

Outputs:

```text
/data/runs/chair_dinov2_test/mesh.obj
/data/runs/chair_dinov2_test/mesh.ply
/data/runs/chair_dinov2_test/udf_grid.npy
```

## 5. About TripoSR Fine-Tuning

TripoSR is the more realistic pretrained base for domain adaptation, but the
official repository primarily exposes inference code, not a clean training API.
The Hugging Face config shows it uses:

```text
DINOSingleImageTokenizer
Triplane1DTokenizer
Transformer1D
TriplaneUpsampleNetwork
NeRFMLP
TriplaneNeRFRenderer
```

Fine-tuning it properly requires adding a training loop around its internal
renderer and supervising rendered RGB/mask/depth/normal from our camera views.
That is a second step after the DINOv2 baseline is confirmed.

Recommended order:

```text
1. Train DINOv2 triplane baseline on the new dataset.
2. Use it to verify data quality and mesh target scale.
3. Then adapt TripoSR internals with a small LR and frozen image tokenizer.
```

