"""에이전트별 LLM 모델 설정.

.env에서 에이전트별 모델을 바꿔가며 A/B 테스트할 수 있다:

    AI_MODEL_DESIGN=gpt-4o-mini       # 설계 에이전트
    AI_MODEL_DEVELOP=gpt-5.6-luna     # 개발 에이전트 (Terraform 코드 생성)
    AI_MODEL_VERIFY=gpt-5.6-luna      # 검증 에이전트 (보안/최적화 LLM 분석)
    AI_MODEL_RAG=gpt-4o-mini          # RAG 쿼리 확장 (번역 + multi-query)
    AI_REASONING_EFFORT=low           # reasoning 모델(gpt-5*/o*)에만 적용

미설정 시 기본값은 gpt-4o-mini.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-4o-mini"

# reasoning 모델 판별용 접두사 (temperature 미지원, reasoning_effort 지원)
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(_REASONING_PREFIXES)


def get_llm(agent: str, temperature: float = 0.0) -> ChatOpenAI:
    """에이전트 이름(design/develop/verify/rag)에 해당하는 LLM 인스턴스를 만든다.

    - 일반 모델(gpt-4o 계열): temperature 적용 (기본 0 — 결정적 출력)
    - reasoning 모델(gpt-5/o 계열): temperature 미지원이므로 생략하고
      AI_REASONING_EFFORT(기본 low)를 적용 — 추론 토큰 과금/지연 통제
    """
    model = os.environ.get(f"AI_MODEL_{agent.upper()}", DEFAULT_MODEL)

    # 요청이 무한정 매달리지 않도록 타임아웃/재시도 상한 (호출 1회 최대 ~5분)
    common = {"timeout": 120, "max_retries": 2}

    if _is_reasoning_model(model):
        effort = os.environ.get("AI_REASONING_EFFORT", "low")
        return ChatOpenAI(model=model, reasoning_effort=effort, **common)

    return ChatOpenAI(model=model, temperature=temperature, **common)
