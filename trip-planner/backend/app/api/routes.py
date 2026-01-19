# genAI_master_start
from fastapi import APIRouter

from app.api.v1.chat import router as chat_router
from app.schemas.common import HealthResponse

router = APIRouter()
router.include_router(chat_router, prefix="/v1")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")
# genAI_master_end
