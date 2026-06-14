from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.orm import Session

from backend.database.config import get_db
from backend.schema import user_schema
from backend.service import user_service

router = APIRouter(
    prefix="/api/user",
)

@router.post("/create")
def user_create(_user_create: user_schema.UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = user_service.get_user_by_email(db, _user_create.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 등록된 이메일 주소입니다."
        )
    user_service.create_user(db=db, user_create=_user_create)
    return {"message": "회원가입이 성공적으로 완료되었습니다."}

@router.post("/login")
def user_login(
    login_data: user_schema.UserLogin,
    response: Response,
    db: Session = Depends(get_db)
):
    user = user_service.get_user_by_email(db, login_data.email)
    if not user or not user_service.verify_password(login_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이메일 또는 비밀번호가 올바르지 않습니다."
        )
    
    # Generate token
    token = user_service.create_token({"sub": user.email})
    
    # Set httpOnly cookie
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=86400 * 7, # 7 days
        expires=86400 * 7,
        samesite="lax",
        secure=False, # Set to True in HTTPS production
    )
    return {"message": "로그인이 성공적으로 완료되었습니다."}

@router.post("/logout")
def user_logout(response: Response):
    response.delete_cookie(
        key="access_token",
        samesite="lax",
        httponly=True
    )
    return {"message": "로그아웃 되었습니다."}

@router.get("/me", response_model=user_schema.UserResponse)
def get_me(
    current_user = Depends(user_service.get_current_user)
):
    return current_user
