from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from authlib.integrations.starlette_client import OAuthError

from backend.database.config import get_db
from backend.schema import user_schema
from backend.service import user_service
from backend.service import oauth_service

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
    
    # Generate token + set httpOnly cookie
    token = user_service.create_token({"sub": user.email})
    user_service.set_auth_cookie(response, token)
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


# ─── 소셜 로그인 (OAuth: github, google) ──────────────────────────────────────

@router.get("/oauth/{provider}/login")
async def oauth_login(provider: str, request: Request):
    """제공자 인증 페이지로 리다이렉트한다."""
    if provider not in oauth_service.SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="지원하지 않는 제공자입니다.")
    if not oauth_service.is_configured(provider):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{provider} 로그인이 서버에 설정되지 않았습니다. (CLIENT_ID/SECRET 필요)",
        )
    client = oauth_service.oauth.create_client(provider)
    return await client.authorize_redirect(request, oauth_service.redirect_uri(provider))


@router.get("/oauth/{provider}/callback")
async def oauth_callback(provider: str, request: Request, db: Session = Depends(get_db)):
    """OAuth 콜백: 코드→토큰 교환→이메일 조회→유저 find-or-create→JWT 쿠키→프론트 리다이렉트."""
    frontend = oauth_service.FRONTEND_URL
    if provider not in oauth_service.SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="지원하지 않는 제공자입니다.")

    client = oauth_service.oauth.create_client(provider)
    try:
        token = await client.authorize_access_token(request)
    except OAuthError:
        return RedirectResponse(f"{frontend}/?auth_error=oauth")

    email = await oauth_service.fetch_email(provider, token)
    if not email:
        return RedirectResponse(f"{frontend}/?auth_error=email")

    user = user_service.get_or_create_oauth_user(db, email, provider)
    jwt_token = user_service.create_token({"sub": user.email})

    redirect = RedirectResponse(frontend)
    user_service.set_auth_cookie(redirect, jwt_token)
    return redirect
