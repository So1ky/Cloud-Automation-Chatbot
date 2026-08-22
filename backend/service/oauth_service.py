"""소셜 로그인(OAuth) 설정 — authlib 기반.

현재 GitHub, Google을 지원한다. 제공자별 CLIENT_ID/SECRET는 .env에서 읽으며,
미설정 시에도 서버는 정상 기동하고 해당 OAuth 엔드포인트 호출 시에만 에러를 낸다.
"""

import os

from authlib.integrations.starlette_client import OAuth

SUPPORTED_PROVIDERS = ("github", "google", "kakao")

# client_secret이 반드시 필요한 제공자 (kakao는 REST API 키만으로 동작, secret 선택)
_SECRET_REQUIRED = ("github", "google")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# 제공자별 콜백 URL (환경변수로 재정의 가능, 기본은 로컬)
_REDIRECT_URIS = {
    "github": os.getenv(
        "GITHUB_REDIRECT_URI",
        "http://localhost:8000/api/user/oauth/github/callback",
    ),
    "google": os.getenv(
        "GOOGLE_REDIRECT_URI",
        "http://localhost:8000/api/user/oauth/google/callback",
    ),
    "kakao": os.getenv(
        "KAKAO_REDIRECT_URI",
        "http://localhost:8000/api/user/oauth/kakao/callback",
    ),
}

oauth = OAuth()

oauth.register(
    name="github",
    client_id=os.getenv("GITHUB_CLIENT_ID"),
    client_secret=os.getenv("GITHUB_CLIENT_SECRET"),
    access_token_url="https://github.com/login/oauth/access_token",
    authorize_url="https://github.com/login/oauth/authorize",
    api_base_url="https://api.github.com/",
    client_kwargs={"scope": "read:user user:email"},
)

# Google은 OpenID Connect — discovery 문서로 authorize/token/userinfo를 자동 구성
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# Kakao — OAuth2. client_id는 REST API 키, client_secret은 선택(보안 설정에서 활성화 시).
oauth.register(
    name="kakao",
    client_id=os.getenv("KAKAO_CLIENT_ID"),
    client_secret=os.getenv("KAKAO_CLIENT_SECRET") or None,
    access_token_url="https://kauth.kakao.com/oauth/token",
    authorize_url="https://kauth.kakao.com/oauth/authorize",
    api_base_url="https://kapi.kakao.com/",
    client_kwargs={"scope": "account_email"},
)


def redirect_uri(provider: str) -> str:
    return _REDIRECT_URIS[provider]


def is_configured(provider: str) -> bool:
    """해당 제공자가 사용 가능하게 설정되어 있는지.

    github/google은 client_secret도 필수, kakao는 REST API 키(client_id)만 있으면 된다.
    """
    client = oauth.create_client(provider)
    if not client or not client.client_id:
        return False
    if provider in _SECRET_REQUIRED and not client.client_secret:
        return False
    return True


async def fetch_email(provider: str, token: dict) -> str | None:
    """토큰으로 제공자에서 사용자 이메일을 얻는다 (제공자별 방식 차이 흡수)."""
    if provider == "google":
        # OIDC: id_token 파싱 결과가 token["userinfo"]에 담긴다 (없으면 userinfo 엔드포인트)
        info = token.get("userinfo")
        if not info:
            info = (await oauth.google.userinfo(token=token)) or {}
        if info.get("email_verified") is False:
            return None
        return info.get("email")

    if provider == "github":
        profile = (await oauth.github.get("user", token=token)).json()
        email = profile.get("email")
        if not email:
            emails = (await oauth.github.get("user/emails", token=token)).json()
            if isinstance(emails, list):
                email = next(
                    (e["email"] for e in emails if e.get("primary") and e.get("verified")),
                    None,
                ) or next((e["email"] for e in emails if e.get("verified")), None)
        return email

    if provider == "kakao":
        # https://kapi.kakao.com/v2/user/me → kakao_account.email
        resp = (await oauth.kakao.get("v2/user/me", token=token)).json()
        account = resp.get("kakao_account", {}) if isinstance(resp, dict) else {}
        if account.get("is_email_valid") is False or account.get("is_email_verified") is False:
            return None
        return account.get("email")

    return None
