import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from .routers.infer import router as infer_router
from .routers.v1 import router as v1_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="CV Detector/Segmentator API",
        version="0.1.0",
        description="MVP: детектор/сегментатор (11 инструментов), JSON-выгрузка, пороги, сигнал ручного пересчёта",
    )

    allowed_origins = os.getenv("CORS_ALLOW_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173,*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in allowed_origins if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    # static serving for saved outputs (JSON/PNG)
    if os.path.isdir("outputs"):
        app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

    app.include_router(infer_router, prefix="/api")
    app.include_router(v1_router)
    return app


app = create_app()


