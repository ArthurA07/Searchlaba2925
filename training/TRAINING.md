# Обучение: train≈infer, ROI→2048

## Подготовка датасета
1) Инжест (EXIF, ROI, split, symlink):
```bash
python3 tools/ingest_images.py \
  --input /placeholder \
  --output /path/to/dataset_out
```
(при наличии `.env.local` — пути берутся оттуда; одиночные — из папок 1..11, группы — из `$GROUP_DIR`, линейка — исключена)

2) Конвертация разметки (Label Studio/CVAT → YOLO‑Seg) в split:
```bash
# Label Studio JSON
python3 tools/convert_labels.py ls \
  --input /path/to/ls_export.json \
  --dataset-root /path/to/dataset_out

# CVAT JSON
python3 tools/convert_labels.py cvat \
  --input /path/to/cvat.json \
  --dataset-root /path/to/dataset_out
```

3) (Опционально) ROI‑кропы для части train:
```bash
python3 tools/make_train_crops.py --dataset-root /path/to/dataset_out --ratio 0.5
```

## Тренировка
- Базовый этап @1024: `training/train_1024.sh`
- Дофайнтюн @2048: `training/finetune_2048.sh`
- Экспорт ONNX и подключение к API: `training/export_onnx.sh`

ONNX будет сохранён в `backend/models/best.onnx`; API подхватит его через `MODEL_PATH`.

## Замечания
- Параметры аугментаций можно корректировать под датасет.
- Проверяйте оверлеи: `tools/verify_annotations.py --images-single dataset/val/images --images-group dataset/val/group --labels dataset/labels/val --out out/verify`.
