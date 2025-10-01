# CV MVP: YOLOv8s-seg + FastAPI + React (Docker-ready)

## Требования
- Python 3.10+ (используем venv)
- Node.js 20 + npm
- (Опционально) Docker Desktop

## Быстрый старт (локально)
### Backend (FastAPI)
```bash
cd "./backend"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
cd ..
export USE_YOLO=1 YOLO_MODEL=yolov8s-seg.pt NUM_TOOLS=11
PYTHONPATH=backend uvicorn app.main:app --host 127.0.0.1 --port 8000
# Swagger: http://127.0.0.1:8000/docs
```

### Frontend (React + Vite)
```bash
cd ./frontend
npm install
VITE_API_URL=http://127.0.0.1:8000/api npm run dev -- --host 127.0.0.1
# UI: http://127.0.0.1:5173
```

## Docker
```bash
# после установки Docker
docker compose build backend
docker compose up -d backend
# Swagger: http://127.0.0.1:8000/docs
```
- Путь к весам меняется через `YOLO_MODEL` (можно указать локальный .pt).

## Формат JSON ответа /api/infer
```json
{
  "tools": [{"id": 1, "present": true, "score": 0.83}, ...],
  "manual_recount": false,
  "threshold": 0.5,
  "model": "yolo-seg",
  "image_size": [1920, 1080]
}
```

## Инструменты датасета (tools/)
### Инжест изображений
```bash
python3 tools/ingest_images.py \
  --input /path/to/images_root \
  --output /path/to/dataset_out \
  --class-map /path/to/class_map.json \
  --split 0.7 0.2 0.1
# Создаёт originals/, roi/, dataset/{train,val,test}/, manifest.json
```

### Конвертеры разметки → YOLO‑Seg
```bash
# Label Studio
python3 tools/convert_labels.py ls \
  --input /path/to/ls_export.json \
  --out /path/to/yolo_seg/labels \
  --class-map /path/to/ls_class_map.json  # {"tool_1":0,...,"tool_11":10}

# CVAT (JSON)
python3 tools/convert_labels.py cvat \
  --input /path/to/cvat.json \
  --out /path/to/yolo_seg/labels \
  --id-map /path/to/cvat_id_map.json     # {"1":0,...,"11":10}
```

### Верификация разметки
```bash
python3 tools/verify_annotations.py \
  --images /path/to/dataset_out/dataset/val/images \
  --labels /path/to/yolo_seg/labels \
  --out /path/to/out/verify \
  --num 30
```

## Разметка
- Конфиги: `docs/label_studio_config.xml`, `docs/cvat_labels.yaml`
- Гайд: `docs/LABELING_GUIDE.md`
- Классы: 11 инструментов (имена можно заменить на финальные из ТЗ)

## Заметки
- Пересборка Docker не требуется при смене весов: используйте `YOLO_MODEL` или volume.
- `manual_recount` подсвечивает пограничные случаи для ручной проверки.
