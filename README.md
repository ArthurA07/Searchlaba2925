# Hack2025 Detector (GPU, ONNX Runtime)

## Быстрый старт (Docker)
1. Клонируйте репозиторий.
2. Скачайте архив кода (без данных) из  и распакуйте в корень:
   
3. Подготовьте переменные окружения:
   
4. Запуск стека с GPU:
   

## Ключевые ENV
- ENGINE=onnxrt
- MODEL_PATH=/app/models/current.onnx
- ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
- INFER_SIZE=1536
- USE_YOLO=0  # чтобы исключить CPU-ветку

## Проверка

Должно быть .
