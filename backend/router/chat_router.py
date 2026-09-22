from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
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


def _resolve_previous_turn(
    db: Session, conversation_id: int, current_user
) -> ChatHistory:
    """후속 턴 요청 시 이어갈 대화의 직전 턴을 찾고 접근 권한을 검사한다 (404 먼저 → 403)."""
    latest = (
        db.query(ChatHistory)
        .filter(ChatHistory.conversation_id == conversation_id)
        .order_by(ChatHistory.id.desc())
        .first()
    )
    if latest is None:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")

    if latest.user_id is not None:
        if current_user is None or latest.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="이 대화를 이어갈 권한이 없습니다.")

    # 최신 턴에 설계 YAML이 없으면(레거시 데이터) 같은 대화에서 보유한 최신 턴으로 폴백
    if not latest.yaml_output:
        latest = (
            db.query(ChatHistory)
            .filter(
                ChatHistory.conversation_id == conversation_id,
                ChatHistory.yaml_output.isnot(None),
                ChatHistory.yaml_output != "",
            )
            .order_by(ChatHistory.id.desc())
            .first()
        )
    if latest is None or not latest.yaml_output:
        raise HTTPException(
            status_code=400,
            detail="이 대화는 이전 설계 정보가 없어 이어서 수정할 수 없습니다. 새 대화로 요청해 주세요.",
        )
    return latest


@router.post("/")
def handle_chat(
    req: ChatRequest,
    db: Session = Depends(get_db),
    current_user = Depends(user_service.get_current_user_optional)
)-> ChatResponse:
    previous_turn: Optional[ChatHistory] = None
    if req.conversation_id is not None:
        previous_turn = _resolve_previous_turn(db, req.conversation_id, current_user)

    design_result = design(
        req, previous_yaml=previous_turn.yaml_output if previous_turn else None
    )
    base64_image = generate_diagram(design_result.diagram_yaml)

    # 후속 턴은 대화 소유자를 그대로 상속 (한 대화 안에 소유자가 섞이지 않도록)
    if previous_turn is not None:
        user_id = previous_turn.user_id
    else:
        user_id = current_user.id if current_user else None

    chat_history = ChatHistory(
        user_id=user_id,
        conversation_id=req.conversation_id,
        requirements=req.requirements,
        response_message="성공적으로 생성되었습니다.",
        image_url=base64_image,
        terraform_code=design_result.terraform_files,
        validation_summary=design_result.validation_summary,
        cost_estimate=design_result.cost_estimate,
        yaml_output=design_result.yaml_output,
    )
    db.add(chat_history)
    db.flush()  # id 확보 (첫 턴의 conversation_id 백필용)
    if chat_history.conversation_id is None:
        chat_history.conversation_id = chat_history.id
    db.commit()
    db.refresh(chat_history)

    return {
        "status": "success",
        "chat_id": chat_history.id,
        "conversation_id": chat_history.conversation_id,
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
    # 대화 단위로 그룹핑: 첫 턴 행(대표 제목 = 첫 요구사항)을 최근 활동순으로 반환
    grouped = (
        db.query(
            func.min(ChatHistory.id).label("first_id"),
            func.max(ChatHistory.created_at).label("last_at"),
        )
        .filter(ChatHistory.user_id == current_user.id)
        .group_by(ChatHistory.conversation_id)
        .subquery()
    )
    return (
        db.query(ChatHistory)
        .join(grouped, ChatHistory.id == grouped.c.first_id)
        .order_by(grouped.c.last_at.desc())
        .all()
    )


@router.get("/conversation/{conversation_id}", response_model=List[ChatHistoryResponse])
def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(user_service.get_current_user_optional)
):
    turns = (
        db.query(ChatHistory)
        .filter(ChatHistory.conversation_id == conversation_id)
        .order_by(ChatHistory.id.asc())
        .all()
    )
    if not turns:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")

    # 소유권 검사 (대화 내 모든 턴은 동일 user_id — 게스트 대화는 기존 단건 조회와 동일하게 공개)
    if turns[0].user_id is not None:
        if current_user is None or turns[0].user_id != current_user.id:
            raise HTTPException(status_code=403, detail="이 대화 내역을 조회할 권한이 없습니다.")

    return turns


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
