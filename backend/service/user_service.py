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

SECRET_KEY = os.getenv("SECRET_KEY")  

def create_user(db: Session, user_create: UserCreate):
    db_user = User(
        email=user_create.email,
        password=password_hash.hash(user_create.password1)
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_hash.verify(plain_password, hashed_password)

def create_token(payload: dict) -> str:
    # Set expiration to 7 days
    payload = payload.copy()
    payload["exp"] = int(time.time()) + 86400 * 7
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

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
