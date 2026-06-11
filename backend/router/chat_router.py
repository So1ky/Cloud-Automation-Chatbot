from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import logging
from backend.database.config import get_db
from backend.database.models import ChatHistory
from backend.schema.chat_schema import (
    ChatHistoryItemResponse,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
)
from backend.service.design_service import design
from backend.service.diagram_service import generate_diagram

router = APIRouter(
    prefix="/api/chat",
)

# 로깅 설정 (에러 확인용)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.post("/")
def handle_chat(req: ChatRequest, db: Session = Depends(get_db))-> ChatResponse:
    design_result = design(req)
    image_filename = generate_diagram(design_result.diagram_yaml)
    # generate_terraform(design_result.yaml_output)

    chat_history = ChatHistory(
        requirements=req.requirements,
        response_message="성공적으로 생성되었습니다.",
        image_url=f"http://localhost:8000/static/{image_filename}",
    )
    db.add(chat_history)
    db.commit()
    db.refresh(chat_history)

    return {
        "status": "success",
        "chat_id": chat_history.id,
        "image_url": chat_history.image_url,
        "message": chat_history.response_message,
    }


@router.get("/history", response_model=List[ChatHistoryItemResponse])
def get_chat_history(db: Session = Depends(get_db)):
    return (
        db.query(ChatHistory)
        .order_by(ChatHistory.created_at.desc())
        .all()
    )


@router.get("/{chat_id}", response_model=ChatHistoryResponse)
def get_chat_detail(chat_id: int, db: Session = Depends(get_db)):
    chat = db.query(ChatHistory).filter(ChatHistory.id == chat_id).first()
    if chat is None:
        raise HTTPException(status_code=404, detail="채팅 이력을 찾을 수 없습니다.")
    return chat
