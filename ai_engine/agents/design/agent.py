"""설계 에이전트: 사용자 요구사항 + RAG 컨텍스트 → 구조화된 아키텍처 명세 생성."""

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import APIConnectionError, APITimeoutError, RateLimitError

from ai_engine.agents.design.prompts import MULTITURN_INSTRUCTION_TEMPLATE, SYSTEM_PROMPT
from ai_engine.config import get_llm
from ai_engine.rag.knowledge_base import search_knowledge_base
from ai_engine.state.architecture_schema import ArchitectureSpec
from ai_engine.state.graph_state import GraphState


def design_node(state: GraphState) -> dict:
    """LangGraph 노드: RAG 검색 → LLM (structured output) → YAML 변환.

    검증 에이전트가 아키텍처 문제로 되돌려 보낸 경우(feedback 존재)에는
    이전 YAML과 검증 피드백을 함께 전달해 설계를 수정하도록 한다.
    """
    user_requirements = state["user_requirements"]
    feedback = state.get("feedback", "")
    previous_yaml = state.get("yaml_output", "")
    is_retry = bool(feedback) and state.get("fix_target") == "design"
    # 멀티턴: 직전 대화 턴의 확정 설계 (Self-Healing 재시도가 아닐 때만 —
    # 재시도 시엔 방금 실패한 yaml_output이 이미 멀티턴 문맥을 반영하고 있음)
    multiturn_yaml = state.get("previous_yaml", "")
    is_multiturn = bool(multiturn_yaml) and not is_retry

    # 재시도가 아닐 때만 RAG 검색 (재시도 시 기존 컨텍스트 재사용)
    rag_context = state.get("rag_context", "")
    if not is_retry or not rag_context:
        print("[설계 에이전트] RAG 검색 중...")
        try:
            rag_context = search_knowledge_base(user_requirements)
        except Exception as e:
            print(f"[설계 에이전트] RAG 검색 실패: {e}")
            rag_context = "Well-Architected Framework 문서를 검색하지 못했습니다."

    llm = get_llm("design")
    structured_llm = llm.with_structured_output(ArchitectureSpec)

    human_content = (
        f"## AWS Well-Architected Framework 참고 문서\n\n"
        f"{rag_context}\n\n"
        f"---\n\n"
        f"## 사용자 요구사항\n\n{user_requirements}"
    )
    if is_retry:
        print("[설계 에이전트] 검증 피드백 반영하여 재설계 중...")
        human_content += (
            f"\n\n---\n\n"
            f"## 이전 설계 (검증 실패)\n\n{previous_yaml}\n\n"
            f"## 검증 에이전트 피드백\n\n{feedback}\n\n"
            f"## 재설계 지시 (반드시 준수)\n"
            f"1. 피드백에서 지적된 문제를 해결하는 데 필요한 최소한의 변경만 하세요.\n"
            f"2. 문제가 없는 부분은 이전 설계를 그대로 유지하세요.\n"
            f"3. 피드백이 명시적으로 요구하지 않는 한 새로운 AWS 서비스, 도메인, 인증서를 "
            f"추가하지 마세요. 복잡도를 늘리는 방향이 아니라 단순화하는 방향으로 해결하세요."
        )
    elif is_multiturn:
        print("[설계 에이전트] 이전 턴 설계를 반영하여 수정 설계 중...")
        human_content += MULTITURN_INSTRUCTION_TEMPLATE.format(previous_yaml=multiturn_yaml)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    print(f"[설계 에이전트] {llm.model_name} 호출 중...")
    try:
        spec: ArchitectureSpec = structured_llm.invoke(messages)
    except RateLimitError as e:
        raise RuntimeError(f"OpenAI API 요청 한도 초과: {e}") from e
    except APITimeoutError as e:
        raise RuntimeError(f"OpenAI API 타임아웃: {e}") from e
    except APIConnectionError as e:
        raise RuntimeError(f"OpenAI API 연결 실패: {e}") from e
    except Exception as e:
        raise RuntimeError(f"LLM 호출 중 오류 발생: {e}") from e

    spec_dict = spec.model_dump(exclude_none=True)
    yaml_output = yaml.dump(spec_dict, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print("[설계 에이전트] YAML 생성 완료")

    # LangGraph 노드는 변경된 키만 반환한다.
    # messages는 add_messages reducer가 자동으로 기존 히스토리에 append한다.
    return {
        "rag_context": rag_context,
        "yaml_output": yaml_output,
        "messages": [
            HumanMessage(content=user_requirements),
            AIMessage(content=yaml_output),
        ],
    }
