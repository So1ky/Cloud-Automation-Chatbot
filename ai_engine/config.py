"""에이전트별 LLM 모델 설정.

.env에서 에이전트별 모델을 바꿔가며 A/B 테스트할 수 있다:

    AI_MODEL_DESIGN=gpt-4o-mini       # 설계 에이전트
    AI_MODEL_DEVELOP=gpt-5.6-luna     # 개발 에이전트 (Terraform 코드 생성)
    AI_MODEL_VERIFY=gpt-5.6-luna      # 검증 에이전트 (보안/최적화 LLM 분석)
    AI_MODEL_RAG=gpt-4o-mini          # RAG 쿼리 확장 (번역 + multi-query)
    AI_MODEL_DESIGN=claude-opus-5-5   # claude-* 접두사면 Anthropic (ANTHROPIC_API_KEY 필요)
    AI_REASONING_EFFORT=low           # reasoning 모델(gpt-5*/o*/claude-*)에만 적용
    AI_REASONING_EFFORT_DESIGN=high   # 에이전트별 재정의 (미설정 시 AI_REASONING_EFFORT)

미설정 시 기본값은 gpt-4o-mini.
"""

from __future__ import annotations

import json
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, convert_to_messages
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

DEFAULT_MODEL = "gpt-4o-mini"

# reasoning 모델 판별용 접두사 (temperature 미지원, reasoning_effort 지원)
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")
_CLAUDE_PREFIX = "claude-"

# Claude 5.x는 thinking 토큰이 max_tokens에 포함되므로 넉넉히 (모델 상한 128K)
_CLAUDE_MAX_TOKENS = 32000


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(_REASONING_PREFIXES)


def get_reasoning_effort(agent: str) -> str:
    """에이전트별 추론 강도. 에이전트 전용 변수 → 공통 변수 → 기본 low 순으로 결정한다."""
    return os.environ.get(
        f"AI_REASONING_EFFORT_{agent.upper()}",
        os.environ.get("AI_REASONING_EFFORT", "low"),
    )


def _decode_json_strings(value):
    """dict/list 안의 JSON처럼 생긴 문자열 값을 재귀적으로 json.loads 한다 (실패하면 원문 유지)."""
    if isinstance(value, dict):
        return {k: _decode_json_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_json_strings(v) for v in value]
    if isinstance(value, str) and value.lstrip()[:1] in ("{", "["):
        try:
            # strict=False: 문자열 값 안의 줄바꿈 등 제어문자를 허용.
            # raw_decode: Claude가 닫는 중괄호를 하나 더 붙이는 경우("...}}" + "}")가 있어
            # 첫 객체만 읽고 꼬리가 괄호·공백뿐이면 받아들인다.
            obj, end = json.JSONDecoder(strict=False).raw_decode(value.lstrip())
            rest = value.lstrip()[end:].strip()
            if rest and rest.strip("}] ,") != "":
                raise ValueError(f"Extra data: {rest[:40]!r}")
            return _decode_json_strings(obj)
        except ValueError as e:
            print(f"[config] 도구 인자 JSON 문자열 복원 실패: {e} / 앞부분: {value[:120]!r}")
            return value
    return value


def _dump_tool_args(name: str, args) -> None:
    """파싱 실패한 도구 인자 원문을 임시 파일로 남긴다 (원인 분석용, 실패해도 무시)."""
    try:
        import tempfile
        with tempfile.NamedTemporaryFile("w", prefix=f"claude_{name}_", suffix=".json",
                                         delete=False, encoding="utf-8") as f:
            json.dump(args, f, ensure_ascii=False, indent=2)
        print(f"[config] 파싱 실패한 도구 인자 원문 저장: {f.name}")
    except Exception as e:
        print(f"[config] 도구 인자 덤프 실패: {e}")


class ClaudeChat(ChatAnthropic):
    """Claude 5.x용 ChatAnthropic.

    Claude 5.x는 thinking을 끌 수 없고 강제 tool_choice(type=tool/any)를 400으로 거부하므로
    langchain-anthropic 0.3의 기본 with_structured_output(강제 호출)을 쓸 수 없다.
    대신 tool_choice=auto로 스키마를 도구로 붙이고, 시스템 프롬프트에 "반드시 이 도구를 호출하라"를
    덧붙인 뒤 tool call 인자를 Pydantic으로 검증한다. (API 네이티브 json_schema 출력은
    ArchitectureSpec처럼 큰 스키마에서 "compiled grammar is too large"로 실패해 쓰지 않음)
    """

    @property
    def model_name(self) -> str:
        """ChatOpenAI와 같은 속성명으로 로그에서 모델명을 읽을 수 있게 한다."""
        return self.model

    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
        bound = self.bind_tools([schema], tool_choice="auto")
        instruction = (
            f"You MUST respond by calling the `{schema.__name__}` tool exactly once. "
            "Never answer in plain text. Pass nested objects and arrays as real JSON "
            "objects/arrays in the tool input, never as JSON-encoded strings."
        )

        def _add_instruction(messages):
            messages = list(convert_to_messages(messages))
            # Anthropic은 system 메시지가 맨 앞에 연속으로만 올 수 있다 → 선두 system 블록 끝에 삽입
            idx = 0
            while idx < len(messages) and isinstance(messages[idx], SystemMessage):
                idx += 1
            messages.insert(idx, SystemMessage(content=instruction))
            return messages

        def _parse(message):
            if not message.tool_calls:
                raise RuntimeError(f"Claude가 {schema.__name__} 도구를 호출하지 않았습니다")
            args = message.tool_calls[0]["args"]
            try:
                return schema.model_validate(args)
            except ValidationError:
                # Claude가 큰 중첩 객체를 JSON 문자열로 넣는 경우가 있어 한 번 풀어서 재시도
                try:
                    return schema.model_validate(_decode_json_strings(args))
                except ValidationError:
                    _dump_tool_args(schema.__name__, args)
                    raise

        return RunnableLambda(_add_instruction) | bound | RunnableLambda(_parse)


def get_llm(agent: str, temperature: float = 0.0) -> BaseChatModel:
    """에이전트 이름(design/develop/verify/rag)에 해당하는 LLM 인스턴스를 만든다.

    - 일반 모델(gpt-4o 계열): temperature 적용 (기본 0 — 결정적 출력)
    - reasoning 모델(gpt-5/o 계열): temperature 미지원이므로 생략하고
      추론 강도를 적용 — 추론 토큰 과금/지연 통제.
      AI_REASONING_EFFORT_<AGENT>가 있으면 그 값, 없으면 AI_REASONING_EFFORT(기본 low)
    - claude-* : Anthropic. 샘플링 파라미터 미지원이라 temperature 생략, 추론 강도는
      output_config.effort로 전달 (OpenAI의 "none"은 Claude에 없어 low로 매핑)
    """
    model = os.environ.get(f"AI_MODEL_{agent.upper()}", DEFAULT_MODEL)

    # 요청이 무한정 매달리지 않도록 타임아웃/재시도 상한 (호출 1회 최대 ~5분)
    common = {"timeout": 120, "max_retries": 2}

    if model.startswith(_CLAUDE_PREFIX):
        effort = get_reasoning_effort(agent)
        effort = "low" if effort == "none" else effort
        return ClaudeChat(
            model=model,
            max_tokens=_CLAUDE_MAX_TOKENS,
            model_kwargs={"output_config": {"effort": effort}},
            **common,
        )

    if _is_reasoning_model(model):
        return ChatOpenAI(model=model, reasoning_effort=get_reasoning_effort(agent), **common)

    return ChatOpenAI(model=model, temperature=temperature, **common)
