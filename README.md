# Hack2025 Detector (GPU, ONNX Runtime)

Полный рабочий стенд распознавания инструментов на YOLOv8‑seg с инференсом через ONNX Runtime (CUDA).

## Требования
- Docker + Docker Compose
- Хост с GPU NVIDIA (драйвер 535+), установлен `nvidia-container-toolkit`
- Интернет для скачивания зависимостей и LFS‑объектов

## Важно про Git LFS
- Модели (`models/*.onnx`, `models/*.pt`) хранятся в Git LFS.
- КНОПКА “Download ZIP” НЕ скачивает реальные веса (получите 133‑байтовые указатели).
- Правильный способ: `git clone` + `git lfs pull` (см. ниже) или скачайте архив из `releases/`.

## Быстрый старт (рекомендуется: git clone + LFS)
1. Клонируйте репозиторий и подтяните LFS‑артефакты (модели):
   ```bash
   git clone https://github.com/ArthurA07/Searchlaba2925.git
   cd Searchlaba2925
   git lfs install
   git lfs pull
   ```
2. Подготовьте переменные окружения:
   ```bash
   cp backend/.env.example backend/.env
   # при желании откорректируйте пороги/размер инференса
   ```
3. Запустите стек (GPU):
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
   # API: http://localhost:18000/docs
   ```

## Альтернатива (одним файлом, без LFS)
- В репозитории лежит архив `releases/hack2025_dist_code.tgz` (~80 МБ) с кодом и моделями.
- Распаковка в корень проекта:
  ```bash
  tar -xzf releases/hack2025_dist_code.tgz -C .
  rsync -a dist_import/ ./
  cp backend/.env.example backend/.env
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
  ```

## Работа с публичным сервером (без локального развёртывания)
- Swagger/UI: `http://89.108.118.166:18000/docs`
- Проверка движка (ожидаем `engine: onnxrt`, CUDA активен):
  ```bash
  curl -s http://89.108.118.166:18000/api/v1/bench | jq
  ```
- Инференс (пример без ROI, порог 0.40):
  ```bash
  curl -sS -X POST 'http://89.108.118.166:18000/api/v1/infer?score_thr=0.40&roi=false' \
    -F 'file=@/path/to/any.jpg' | jq '{engine, used_imgsz, infer_ms, n_det: (.detections|length)}'
  ```

## Проверка
- Health:
  ```bash
  curl -s http://localhost:18000/health
  ```
- Бенч (должно быть `engine: "onnxrt"`):
  ```bash
  curl -s http://localhost:18000/api/v1/bench | jq
  ```
- Смоук‑инференс (без ROI, с порогом 0.40):
  ```bash
  IMG=./frontend/index.html # замените на любой .jpg/.png
  curl -sS -X POST 'http://localhost:18000/api/v1/infer?score_thr=0.40&roi=false' \
    -F "file=@$IMG" | jq '{engine, used_imgsz, infer_ms, n_det: (.detections|length)}'
  ```

## Важные переменные окружения (`backend/.env`)
- `ENGINE=onnxrt` — использовать ONNX Runtime
- `MODEL_PATH=/app/models/current.onnx` — активная модель
- `ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider` — сначала CUDA
- `INFER_SIZE=1536` — размер инференса (кратно 32)
- `USE_YOLO=0` — не уходить в CPU‑ветку `/api/infer`
- `ROI_ENABLE=true` — включение ROI (обрезка стола)
- `SCORE_THR=0.30`, `IOU_THR=0.50`, `NUM_CLASSES=11`

## Где лежат модели
- `models/best.onnx`, `models/current.onnx`, `models/best.pt`, `models/stage2_best.pt`
- Хранятся под Git LFS; альтернативно — внутри `releases/hack2025_dist_code.tgz`

## Эндпоинты
- `POST /api/v1/infer` — основной инференс (GPU/ONNX)
- `GET  /api/v1/bench` — быстрая проверка движка/провайдеров
- Swagger: `http://localhost:18000/docs`

## Диагностика
- Провайдеры ORT в контейнере:
  ```bash
  docker compose exec backend python - <<'PY'
  import onnxruntime as ort; print(ort.get_available_providers())
  PY
  ```
- Если видите только `CPUExecutionProvider`:
  1) Проверьте, что стек поднят с `docker-compose.gpu.yml`
  2) На хосте установлен `nvidia-container-toolkit`
  3) Драйвер актуален (`nvidia-smi`)

## Примечания
- Первый запрос может быть дольше (прогрев модели на GPU). Последующие — быстрее.
- Для воспроизводимости тренировки см. `training/`.
