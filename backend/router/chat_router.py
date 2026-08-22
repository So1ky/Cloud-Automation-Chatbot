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
from backend.service import user_service

router = APIRouter(
    prefix="/api/chat",
)

# 로깅 설정 (에러 확인용)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.post("/")
def handle_chat(
    req: ChatRequest,
    db: Session = Depends(get_db),
    current_user = Depends(user_service.get_current_user_optional)
)-> ChatResponse:
    design_result = design(req)
    base64_image = generate_diagram(design_result.diagram_yaml)

    user_id = current_user.id if current_user else None

    chat_history = ChatHistory(
        user_id=user_id,
        requirements=req.requirements,
        response_message="성공적으로 생성되었습니다.",
        image_url=base64_image,
        terraform_code=design_result.terraform_files,
        validation_summary=design_result.validation_summary,
        cost_estimate=design_result.cost_estimate,
    )
    db.add(chat_history)
    db.commit()
    db.refresh(chat_history)

    return {
        "status": "success",
        "chat_id": chat_history.id,
        "image_url": chat_history.image_url,
        "message": chat_history.response_message,
        "terraform_code": chat_history.terraform_code,
        "validation_summary": chat_history.validation_summary,
        "cost_estimate": chat_history.cost_estimate,
    }


@router.get("/history", response_model=List[ChatHistoryItemResponse])
def get_chat_history(
    db: Session = Depends(get_db),
    current_user = Depends(user_service.get_current_user)
):
    return (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == current_user.id)
        .order_by(ChatHistory.created_at.desc())
        .all()
    )


@router.get("/{chat_id}", response_model=ChatHistoryResponse)
def get_chat_detail(
    chat_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(user_service.get_current_user_optional)
):
    chat = db.query(ChatHistory).filter(ChatHistory.id == chat_id).first()
    if chat is None:
        raise HTTPException(status_code=404, detail="채팅 이력을 찾을 수 없습니다.")
        
    # 만약 소유주가 지정되어 있는 채팅인데, 현재 로그인된 유저와 다르다면 조회 금지
    if chat.user_id is not None:
        if current_user is None or chat.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="이 채팅 내역을 조회할 권한이 없습니다.")
            
    return chat
