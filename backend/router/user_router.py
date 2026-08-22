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


# ─── 소셜 로그인 (GitHub OAuth) ───────────────────────────────────────────────

@router.get("/oauth/github/login")
async def github_login(request: Request):
    """GitHub 인증 페이지로 리다이렉트한다."""
    if not oauth_service.is_configured("github"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GitHub 로그인이 서버에 설정되지 않았습니다. (GITHUB_CLIENT_ID/SECRET 필요)",
        )
    return await oauth_service.oauth.github.authorize_redirect(
        request, oauth_service.GITHUB_REDIRECT_URI
    )


@router.get("/oauth/github/callback")
async def github_callback(request: Request, db: Session = Depends(get_db)):
    """GitHub 콜백: 코드→토큰 교환→이메일 조회→유저 find-or-create→JWT 쿠키→프론트 리다이렉트."""
    frontend = oauth_service.FRONTEND_URL
    try:
        token = await oauth_service.oauth.github.authorize_access_token(request)
    except OAuthError:
        return RedirectResponse(f"{frontend}/?auth_error=oauth")

    # 프로필 이메일 확보 (공개 이메일이 없으면 /user/emails에서 인증된 primary 선택)
    profile = (await oauth_service.oauth.github.get("user", token=token)).json()
    email = profile.get("email")
    if not email:
        emails = (await oauth_service.oauth.github.get("user/emails", token=token)).json()
        if isinstance(emails, list):
            email = next(
                (e["email"] for e in emails if e.get("primary") and e.get("verified")),
                None,
            ) or next((e["email"] for e in emails if e.get("verified")), None)

    if not email:
        return RedirectResponse(f"{frontend}/?auth_error=email")

    user = user_service.get_or_create_oauth_user(db, email, "github")
    jwt_token = user_service.create_token({"sub": user.email})

    redirect = RedirectResponse(frontend)
    user_service.set_auth_cookie(redirect, jwt_token)
    return redirect
