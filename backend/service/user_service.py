import base64
import hashlib
import hmac
import json
import logging
import os
import time
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status, Request

from backend.schema.user_schema import UserCreate
from backend.database.models import User
from backend.database.config import get_db

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

SECRET_KEY = os.getenv("SECRET_KEY")  

def create_user(db: Session, user_create: UserCreate):
    db_user = User(
        email=user_create.email,
        password=pwd_context.hash(user_create.password1)
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_token(payload: dict) -> str:
    # Set expiration to 7 days
    payload = payload.copy()
    payload["exp"] = time.time() + 86400 * 7
    payload_json = json.dumps(payload)
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode().rstrip("=")
    
    # Calculate HMAC-SHA256 signature
    signature = hmac.new(SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")
    return f"{payload_b64}.{sig_b64}"

def verify_token(token: str) -> dict | None:
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_b64, sig_b64 = parts
        
        # Verify signature
        expected_sig = hmac.new(SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256).digest()
        expected_sig_b64 = base64.urlsafe_b64encode(expected_sig).decode().rstrip("=")
        
        if not hmac.compare_digest(sig_b64, expected_sig_b64):
            return None
            
        # Decode payload
        rem = len(payload_b64) % 4
        if rem > 0:
            payload_b64 += "=" * (4 - rem)
        payload_json = base64.urlsafe_b64decode(payload_b64.encode()).decode()
        payload = json.loads(payload_json)
        
        # Check expiration
        if time.time() > payload.get("exp", 0):
            return None
            
        return payload
    except Exception as e:
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
