# Chair Geometry Baseline

This is a fresh, self-contained Kaggle baseline. It does not import old project
training files.

The first reliable target is a soft surface-proximity field, not true inside/outside
occupancy:

```text
single RGB + fixed mask
-> CNN encoder
-> triplane features
-> MLP field decoder
-> Marching Cubes mesh
```

This is the right sanity path because the current dataset already has surface
points, while true occupancy from arbitrary Objaverse meshes can be fragile.

## 1. Check paths

```bash
!python /kaggle/working/repositoryi/LastChange/chair_geometry_baseline.py \
  --mode check \
  --dataset_root /kaggle/input/datasets/neixon/objaverse-chair-blender-dataset \
  --mask_root /kaggle/working/repaired_masks \
  --clean_uids /kaggle/working/chair_dataset_audit_repaired/clean_uids.txt \
  --splits_json /kaggle/working/chair_dataset_audit_repaired/splits.json
```

Expected rows:

```text
train_rows ~= 371 * 24
val_rows   ~= 46 * 24
test_rows  ~= 47 * 24
```

## 2. Overfit 1 object

Run this before any real training. If it cannot overfit one chair, the model or
data pipeline is wrong.

```bash
!python /kaggle/working/repositoryi/LastChange/chair_geometry_baseline.py \
  --mode train \
  --dataset_root /kaggle/input/datasets/neixon/objaverse-chair-blender-dataset \
  --mask_root /kaggle/working/repaired_masks \
  --clean_uids /kaggle/working/chair_dataset_audit_repaired/clean_uids.txt \
  --splits_json /kaggle/working/chair_dataset_audit_repaired/splits.json \
  --work_dir /kaggle/working/chair_geo_overfit_1 \
  --overfit_objects 1 \
  --epochs 80 \
  --batch_size 4 \
  --steps_per_epoch 50 \
  --queries 4096 \
  --plane_size 64 \
  --plane_channels 32
```

## 3. Overfit 8 objects

```bash
!python /kaggle/working/repositoryi/LastChange/chair_geometry_baseline.py \
  --mode train \
  --dataset_root /kaggle/input/datasets/neixon/objaverse-chair-blender-dataset \
  --mask_root /kaggle/working/repaired_masks \
  --clean_uids /kaggle/working/chair_dataset_audit_repaired/clean_uids.txt \
  --splits_json /kaggle/working/chair_dataset_audit_repaired/splits.json \
  --work_dir /kaggle/working/chair_geo_overfit_8 \
  --overfit_objects 8 \
  --epochs 60 \
  --batch_size 8 \
  --steps_per_epoch 80 \
  --queries 4096
```

## 4. First full train

```bash
!python /kaggle/working/repositoryi/LastChange/chair_geometry_baseline.py \
  --mode train \
  --dataset_root /kaggle/input/datasets/neixon/objaverse-chair-blender-dataset \
  --mask_root /kaggle/working/repaired_masks \
  --clean_uids /kaggle/working/chair_dataset_audit_repaired/clean_uids.txt \
  --splits_json /kaggle/working/chair_dataset_audit_repaired/splits.json \
  --work_dir /kaggle/working/chair_geo_baseline_v1 \
  --epochs 40 \
  --batch_size 8 \
  --queries 4096 \
  --plane_size 64 \
  --plane_channels 32
```

## 5. Extract a mesh

Pick an RGB/mask pair from the fixed dataset and run:

```bash
!python /kaggle/working/repositoryi/LastChange/chair_geometry_baseline.py \
  --mode predict \
  --checkpoint /kaggle/working/chair_geo_overfit_1/best.pt \
  --image /kaggle/input/datasets/neixon/objaverse-chair-blender-dataset/renders/UID/view_000.png \
  --mask /kaggle/working/repaired_masks/UID/view_000.png \
  --output_dir /kaggle/working/chair_geo_prediction \
  --grid_resolution 96 \
  --mc_level 0.35
```

Outputs:

```text
field.npy
mesh.obj
mesh.ply
```

## What good looks like

For `--overfit_objects 1`, train loss should fall clearly and the extracted mesh
should resemble the chosen chair. It does not need to be perfect. This baseline
is only the first controlled test before moving to stronger geometry objectives.
