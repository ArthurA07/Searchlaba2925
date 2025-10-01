import io
from typing import List

from fastapi import APIRouter, File, UploadFile, Query
from PIL import Image

from ..schemas import InferenceResponse, ToolResult
from ..services.engine import get_engine


router = APIRouter(tags=["inference"])


@router.post("/infer", response_model=InferenceResponse)
async def infer(
    file: UploadFile = File(..., description="Фото сверху"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Порог детекции"),
) -> InferenceResponse:
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")

    engine = get_engine()
    results = engine.predict(image, threshold=threshold)

    tools: List[ToolResult] = [
        ToolResult(id=r["id"], present=r["present"], score=r["score"]) for r in results
    ]

    manual_recount = any((tr.present and tr.score < threshold + 0.1) for tr in tools)

    return InferenceResponse(
        tools=tools,
        manual_recount=manual_recount,
        threshold=threshold,
        model=engine.name,
        image_size=image.size,
    )


