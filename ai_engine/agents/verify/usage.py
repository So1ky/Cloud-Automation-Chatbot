"""사용량 기반 과금 리소스의 비용 추정을 위한 사용량 가정 생성.

Infracost는 사용량 정보가 없으면 Lambda·API Gateway·DynamoDB 온디맨드·S3·Cognito 같은
리소스를 0으로 계산한다. 요구사항에 서비스 규모(월 사용자 수)가 있으면 고정 가정으로
리소스별 사용량을 만들어 Infracost usage 파일로 넘긴다.

LLM은 월 사용자 수만 추출하고, 나머지 가정은 아래 상수로 고정한다 (같은 입력 → 같은 금액).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ai_engine.agents.verify.prompts import USAGE_SCALE_PROMPT
from ai_engine.config import get_llm

# ─── 고정 가정 ───────────────────────────────────────────────────────
REQUESTS_PER_USER_PER_DAY = 20        # 1인당 하루 API 요청(메시지) 수
LAMBDA_AVG_DURATION_MS = 200          # 요청당 Lambda 평균 실행 시간
DYNAMODB_READS_PER_REQUEST = 1.0      # 요청당 DynamoDB 읽기 단위
DYNAMODB_WRITES_PER_REQUEST = 0.3     # 요청당 DynamoDB 쓰기 단위
DYNAMODB_STORAGE_MB_PER_USER = 1      # 사용자당 DynamoDB 저장 용량
S3_STORAGE_MB_PER_USER = 10           # 사용자당 S3 저장 용량
WEBSOCKET_CONNECTION_MIN_PER_USER_PER_DAY = 10  # 1인당 하루 WebSocket 연결 시간(분)
DAYS_PER_MONTH = 30

_RESOURCE_RE = re.compile(r'^\s*resource\s+"(aws_\w+)"\s+"(\w+)"', re.MULTILINE)


class UsageScale(BaseModel):
    """요구사항에서 추출한 서비스 규모."""

    monthly_active_users: Optional[int] = Field(
        description="Monthly active users stated in the requirements, or null if no user scale is stated"
    )


@lru_cache(maxsize=64)
def _extract_cached(user_requirements: str) -> Optional[int]:
    llm = get_llm("report")
    result: UsageScale = llm.with_structured_output(UsageScale).invoke([
        SystemMessage(content=USAGE_SCALE_PROMPT),
        HumanMessage(content=user_requirements),
    ])
    mau = result.monthly_active_users
    return mau if mau and mau > 0 else None


def extract_monthly_users(user_requirements: str) -> Optional[int]:
    """요구사항에서 월 사용자 수를 추출한다. 언급이 없거나 추출에 실패하면 None.

    부가 분석이므로 예외를 삼킨다. Self-Healing 라운드마다 재호출되지 않게 성공 결과만 캐시된다.
    """
    if not user_requirements.strip():
        return None
    try:
        return _extract_cached(user_requirements)
    except Exception as e:
        print(f"[검증 에이전트] 서비스 규모 추출 실패 → 사용량 추정 생략: {e}")
        return None


def build_usage(terraform_files: dict, monthly_users: int) -> Tuple[dict, List[str]]:
    """Terraform 리소스별 Infracost 사용량과, 화면에 보여줄 가정 문구를 만든다.

    Returns:
        (resource_usage, notes) — resource_usage는 {"<type>.<name>": {<usage 키>: 값}}.
        사용량 과금 대상 리소스가 없으면 ({}, []).
    """
    by_type: dict = {}
    for content in terraform_files.values():
        for rtype, name in _RESOURCE_RE.findall(content or ""):
            by_type.setdefault(rtype, []).append(f"{rtype}.{name}")

    monthly_requests = monthly_users * REQUESTS_PER_USER_PER_DAY * DAYS_PER_MONTH
    usage: dict = {}
    notes: List[str] = []

    def share(total: float, count: int) -> int:
        """같은 종류 리소스가 여러 개면 총량을 나눠 중복 계산을 막는다."""
        return int(total / count)

    apis = by_type.get("aws_apigatewayv2_api", []) + by_type.get("aws_api_gateway_rest_api", [])
    for addr in apis:
        per_api = share(monthly_requests, len(apis))
        # HTTP API는 monthly_requests, WebSocket API는 monthly_messages/connection_mins만 사용된다
        usage[addr] = {
            "monthly_requests": per_api,
            "monthly_messages": per_api,
            "monthly_connection_mins": share(
                monthly_users * WEBSOCKET_CONNECTION_MIN_PER_USER_PER_DAY * DAYS_PER_MONTH, len(apis)
            ),
        }

    lambdas = by_type.get("aws_lambda_function", [])
    for addr in lambdas:
        usage[addr] = {
            "monthly_requests": share(monthly_requests, len(lambdas)),
            "request_duration_ms": LAMBDA_AVG_DURATION_MS,
        }

    tables = by_type.get("aws_dynamodb_table", [])
    for addr in tables:
        usage[addr] = {
            "monthly_read_request_units": share(monthly_requests * DYNAMODB_READS_PER_REQUEST, len(tables)),
            "monthly_write_request_units": share(monthly_requests * DYNAMODB_WRITES_PER_REQUEST, len(tables)),
            "storage_gb": share(monthly_users * DYNAMODB_STORAGE_MB_PER_USER / 1024, len(tables)),
        }

    buckets = by_type.get("aws_s3_bucket", [])
    for addr in buckets:
        usage[addr] = {
            "standard": {"storage_gb": share(monthly_users * S3_STORAGE_MB_PER_USER / 1024, len(buckets))},
        }

    for addr in by_type.get("aws_cognito_user_pool", []):
        usage[addr] = {"monthly_active_users": monthly_users}

    if apis or lambdas or tables:
        notes.append(f"1인당 하루 {REQUESTS_PER_USER_PER_DAY}회 요청")
    if lambdas:
        notes.append(f"요청당 Lambda 실행 {LAMBDA_AVG_DURATION_MS}ms")
    if buckets:
        notes.append(f"사용자당 파일 저장 {S3_STORAGE_MB_PER_USER}MB")

    return usage, notes
