"""소셜 로그인(OAuth) 설정 — authlib 기반.

현재 GitHub, Google을 지원한다. 제공자별 CLIENT_ID/SECRET는 .env에서 읽으며,
미설정 시에도 서버는 정상 기동하고 해당 OAuth 엔드포인트 호출 시에만 에러를 낸다.
"""

import os

from authlib.integrations.starlette_client import OAuth

SUPPORTED_PROVIDERS = ("github", "google")

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


def redirect_uri(provider: str) -> str:
    return _REDIRECT_URIS[provider]


def is_configured(provider: str) -> bool:
    """해당 제공자의 CLIENT_ID/SECRET가 설정되어 있는지."""
    client = oauth.create_client(provider)
    return bool(client and client.client_id and client.client_secret)


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

    return None
