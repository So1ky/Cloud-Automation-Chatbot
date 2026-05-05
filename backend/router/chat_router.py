from fastapi import APIRouter
import logging
from backend.schema.chat_schema import ChatRequest, ChatResponse
from backend.service.design_service import design
from backend.service.diagram_service import generate_diagram

router = APIRouter(
    prefix="/api/chat",
)

# 로깅 설정 (에러 확인용)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.post("/")
def handle_chat(req: ChatRequest)-> ChatResponse:
    design_result = design(req)
    image_filename = generate_diagram(design_result.diagram_yaml)
    # generate_terraform(design_result.yaml_output)

    return {
        "status": "success",
        "image_url": f"http://localhost:8000/static/{image_filename}",
        "message": "성공적으로 생성되었습니다."
    }
  
