# Деплой (Docker)

## Backend
```bash
cd backend
# правим .env или экспортируем переменные окружения
# MODEL_PATH, INFER_SIZE=2048, ROI_ENABLE=true, SCORE_THR, IOU_THR, THRESHOLD
cd ..
docker compose build backend
docker compose up -d backend
# http://127.0.0.1:8000/docs
```

## Frontend (опционально)
- Можно собрать образ `frontend` и добавить reverse-proxy (Nginx) для отдачи UI и проксирования API.

## Обновление весов
- Новые веса подключаются через переменную `YOLO_MODEL` (для fallback) или `MODEL_PATH` (ONNX) без пересборки образа.
