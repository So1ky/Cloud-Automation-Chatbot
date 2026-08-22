import logging
import os
import time
import jwt
from pwdlib import PasswordHash
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status, Request

from backend.schema.user_schema import UserCreate
from backend.database.models import User
from backend.database.config import get_db

logger = logging.getLogger(__name__)

password_hash = PasswordHash.recommended()

# JWT 서명 키. 반드시 .env의 SECRET_KEY로 설정해야 한다.
# 미설정 시 임시 키로 넘어가지 않고, 시작 시점에 명확한 에러로 즉시 실패한다.
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY 환경변수가 설정되지 않았습니다. .env에 SECRET_KEY를 설정하세요 "
        '(예: python -c "import secrets; print(secrets.token_urlsafe(48))").'
    )

def create_user(db: Session, user_create: UserCreate):
    db_user = User(
        email=user_create.email,
        password=password_hash.hash(user_create.password1)
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def create_oauth_user(db: Session, email: str, provider: str) -> User:
    """소셜 로그인 유저를 생성한다 (비밀번호 없음)."""
    db_user = User(email=email, password=None, provider=provider)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_or_create_oauth_user(db: Session, email: str, provider: str) -> User:
    """이메일로 기존 유저를 찾고, 없으면 소셜 유저로 생성한다."""
    user = get_user_by_email(db, email)
    if user:
        return user
    return create_oauth_user(db, email, provider)


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_hash.verify(plain_password, hashed_password)

def create_token(payload: dict) -> str:
    # Set expiration to 7 days
    payload = payload.copy()
    payload["exp"] = int(time.time()) + 86400 * 7
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def set_auth_cookie(response, token: str) -> None:
    """JWT를 httpOnly 쿠키로 심는다 (이메일 로그인·소셜 로그인 공용)."""
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=86400 * 7,  # 7 days
        expires=86400 * 7,
        samesite="lax",
        secure=False,  # Set to True in HTTPS production
    )

def verify_token(token: str) -> dict | None:
    try:
        # PyJWT automatically verifies signature and expiration (exp)
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.error(f"Token verification error: {e}")
        return None

def get_current_user_optional(request: Request, db: Session = Depends(get_db)) -> User | None:
    token = request.cookies.get("access_token")
    if not token:
        return None
    
    payload = verify_token(token)
    if not payload:
        return None
        
    email = payload.get("sub")
    if not email:
        return None
        
    return get_user_by_email(db, email)

def get_current_user(user: User | None = Depends(get_current_user_optional)) -> User:
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="로그인이 필요한 서비스입니다.",
        )
    return user
