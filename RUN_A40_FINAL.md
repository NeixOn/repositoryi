# A40 x2 Final Runbook

This setup is for a paid server with 2 x A40, CUDA 12.6, large RAM, and about 100 GB disk.

## 0. Check Machine

```bash
nvidia-smi
df -h
python3 --version
```

Use `/data` for everything if it exists. If not:

```bash
mkdir -p /data
```

## 1. Install Base Packages

```bash
cd /data
git clone https://github.com/NeixOn/repositoryi.git || true
cd /data/repositoryi
git pull

python3 -m pip install --upgrade pip
python3 -m pip install --root-user-action=ignore torch torchvision --index-url https://download.pytorch.org/whl/cu126
python3 -m pip install --root-user-action=ignore pandas tqdm Pillow trimesh scipy scikit-image awscli
```

If the CUDA 12.6 PyTorch index is unavailable, use the normal PyTorch install command from pytorch.org for CUDA 12.x.

## 2. Prepare ABO Chair Dataset

This downloads only chair-like ABO assets, not the full ABO archive.

Start with 1000 objects to verify everything:

```bash
python3 /data/repositoryi/prepare_abo_chairs.py \
  --output_dir /data/abo_chairs \
  --cache_dir /data/abo_cache \
  --num_objects 1000 \
  --max_candidates 12000 \
  --views_per_object 24 \
  --points 65536 \
  --max_total_gb 85
```

If disk still has room and the dataset looks good, increase to 2000-3000:

```bash
python3 /data/repositoryi/prepare_abo_chairs.py \
  --output_dir /data/abo_chairs \
  --cache_dir /data/abo_cache \
  --num_objects 2500 \
  --max_candidates 12000 \
  --views_per_object 24 \
  --points 65536 \
  --max_total_gb 85
```

Expected output:

```text
/data/abo_chairs/metadata/views.csv
/data/abo_chairs/metadata/objects.csv
/data/abo_chairs/objects/<uid>/normalized.glb
/data/abo_chairs/objects/<uid>/points.npz
/data/abo_chairs/renders/<uid>/view_000.png
```

## 3. Train Mesh Model On 2 x A40

Start with the stable configuration:

```bash
torchrun --standalone --nproc_per_node=2 /data/repositoryi/train_chair_triplane_udf_cuda.py \
  --mode train \
  --dataset_root /data/abo_chairs \
  --work_dir /data/runs/chair_triplane_udf \
  --encoder convnext_base \
  --image_size 256 \
  --batch_size 2 \
  --grad_accum 8 \
  --queries_per_item 32768 \
  --surface_points 8192 \
  --plane_size 128 \
  --plane_channels 48 \
  --decoder_hidden 384 \
  --latent_dim 1024 \
  --epochs 80 \
  --lr 1.5e-4 \
  --amp fp16 \
  --num_workers 16 \
  --freeze_encoder_epochs 5 \
  --require_cuda
```

If VRAM is comfortable, stronger run:

```bash
torchrun --standalone --nproc_per_node=2 /data/repositoryi/train_chair_triplane_udf_cuda.py \
  --mode train \
  --dataset_root /data/abo_chairs \
  --work_dir /data/runs/chair_triplane_udf_big \
  --encoder convnext_base \
  --image_size 320 \
  --batch_size 2 \
  --grad_accum 8 \
  --queries_per_item 65536 \
  --surface_points 16384 \
  --plane_size 128 \
  --plane_channels 64 \
  --decoder_hidden 512 \
  --latent_dim 1536 \
  --epochs 120 \
  --lr 1.0e-4 \
  --amp fp16 \
  --num_workers 24 \
  --freeze_encoder_epochs 8 \
  --require_cuda
```

## 4. Test On A Custom Image

```bash
python3 /data/repositoryi/train_chair_triplane_udf_cuda.py \
  --mode predict \
  --checkpoint /data/runs/chair_triplane_udf/best.pt \
  --image /data/my_chair.png \
  --output_dir /data/runs/test_my_chair \
  --grid_resolution 128 \
  --level 0.025 \
  --crop \
  --skip_install
```

Outputs:

```text
*_model_input.png
*_udf_grid.npy
*_mesh.obj
*_mesh.ply
```

## Important Reality Check

This is the strongest practical architecture in this repo for training from your own chair data:

```text
pretrained ConvNeXt image encoder -> triplane feature field -> UDF decoder -> marching cubes mesh
```

It is much stronger than point-cloud regression. But it is still not the same as training TripoSR/Hunyuan3D from scratch. For TripoSR-level quality, the limiting factor is mostly dataset scale and training recipe, not just GPU memory.
