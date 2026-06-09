# Chair Dataset Audit Kit

Этот набор нужен, чтобы на Kaggle быстро понять, годится ли датасет `objaverse_chairs_blender` для обучения сильной модели геометрии стульев.

## Что проверяется

- наличие всех ожидаемых директорий;
- исключение UID из `bad_uids.txt`;
- наличие 24 RGB, mask и camera файлов на каждый объект;
- разрешение `512x512`;
- читаемость изображений;
- пустые или почти пустые RGB;
- валидность масок: foreground ratio, бинарность, количество компонент, касание границ изображения;
- возможность пересоздать маски из RGB без Blender;
- наличие и базовая валидность `normalized.glb`;
- наличие `points.npz`, ключ `points`, форма `(32768, 3)`, finite values, масштаб;
- согласованность камер: шаг азимута около 30 градусов, стабильный radius, верхние 12 видов выше нижних;
- дублирующиеся views по низкоразмерному image hash;
- train/val/test split только по объектам.

## Быстрый запуск в Kaggle

```bash
python /kaggle/working/repositoryi/LastChange/audit_chair_dataset.py \
  --dataset_root /kaggle/input/objaverse-chairs-blender/objaverse_chairs_blender \
  --bad_uids /kaggle/working/repositoryi/bad_uids.txt \
  --out_dir /kaggle/working/chair_dataset_audit \
  --views 24 \
  --resolution 512 \
  --points_per_object 32768 \
  --expected_objects 500
```

Если маски сломаны и надо сразу пересоздать их из RGB:

```bash
python /kaggle/working/repositoryi/LastChange/audit_chair_dataset.py \
  --dataset_root /kaggle/input/objaverse-chairs-blender/objaverse_chairs_blender \
  --bad_uids /kaggle/working/repositoryi/bad_uids.txt \
  --out_dir /kaggle/working/chair_dataset_audit \
  --mask_output_dir /kaggle/working/repaired_masks \
  --repair_masks_from_rgb \
  --repair_force
```

Такой режим не пытается писать в `/kaggle/input`; новые маски окажутся в `/kaggle/working/repaired_masks/<uid>/view_###.png`.

## Выходные файлы

- `summary.json` - главный итог;
- `issues.csv` - все найденные проблемы;
- `objects_audit.csv` - статус каждого UID;
- `views_audit.csv` - метрики каждого view;
- `clean_uids.txt` - UID, которые прошли полный допуск;
- `splits.json` - train/val/test split по объектам;
- `preview_rgb_mask_view000.jpg` - быстрая визуальная проверка RGB/mask.

## Как принимать решение

Датасет можно пускать в обучение, если:

- `usable_objects` близко к `500 - len(bad_uids)`;
- в `issues.csv` нет `severity=block` для нужных UID;
- `preview_rgb_mask_view000.jpg` выглядит нормально;
- `clean_uids.txt` содержит достаточно объектов;
- `splits.json` создан и split сделан по UID, а не по отдельным изображениям.

Для сильной chair-only geometry модели лучше не обучаться на объектах с частичными view, плохими points или неверной камерой. Их стоит держать отдельно и чинить вручную либо исключать.
