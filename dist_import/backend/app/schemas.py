from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    id: int = Field(..., description="Идентификатор инструмента (1..11)")
    present: bool = Field(..., description="Найден ли инструмент")
    score: float = Field(..., ge=0.0, le=1.0, description="Уверенность модели [0..1]")


class InferenceResponse(BaseModel):
    tools: List[ToolResult]
    manual_recount: bool = Field(
        ..., description="Требуется ли ручная проверка/пересчёт"
    )
    threshold: float = Field(..., ge=0.0, le=1.0, description="Порог детекции")
    model: str = Field(..., description="Имя/вариант модели")
    image_size: Optional[Tuple[int, int]] = Field(
        default=None, description="Ширина, высота исходного изображения"
    )


