"""소셜 로그인(OAuth) 설정 — authlib 기반.

현재 GitHub를 지원한다. 제공자별 CLIENT_ID/SECRET는 .env에서 읽으며,
미설정 시에도 서버는 정상 기동하고 해당 OAuth 엔드포인트 호출 시에만 에러를 낸다.
"""

import os

from authlib.integrations.starlette_client import OAuth

# 콜백/프론트 URL (환경변수로 재정의 가능)
GITHUB_REDIRECT_URI = os.getenv(
    "GITHUB_REDIRECT_URI", "http://localhost:8000/api/user/oauth/github/callback"
)
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

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


def is_configured(provider: str) -> bool:
    """해당 제공자의 CLIENT_ID/SECRET가 설정되어 있는지."""
    client = oauth.create_client(provider)
    return bool(client and client.client_id and client.client_secret)
