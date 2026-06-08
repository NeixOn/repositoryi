# newModel

Разбитая версия `train_chair_dinov2_lrm_render_cuda.py`.

## Файлы

- `train.py` - точка входа: читает аргументы и запускает `train()` или `predict()`.
- `args.py` - все аргументы командной строки.
- `trainer.py` - главный цикл обучения, validation, logging, DDP/CUDA setup.
- `data.py` - чтение датасета, split объектов, загрузка RGB/mask, camera rays, `RenderPairDataset`.
- `model.py` - архитектура: DINOv2 encoder, triplane generator, radiance/density decoder.
- `rendering.py` - differentiable volume rendering по camera rays.
- `losses.py` - RGB/mask/geometry/perceptual losses и stage weights.
- `preview.py` - сохранение validation preview.
- `predict.py` - инференс по одному изображению и extraction mesh через marching cubes.
- `checkpoints.py` - сохранение и загрузка checkpoints.
- `constants.py` - общий 3D bounding box.
- `deps.py` - установка зависимостей, если не указан `--skip_install`.

## Запуск

Из корня репозитория или по абсолютному пути:

```bash
python3 ~/repositoryi/newModel/train.py --mode train ...
```

Аргументы сохранены такими же, как у старого файла.
