"""LangGraph 그래프 정의: 설계 → 개발 → 검증 에이전트 워크플로우 (Self-Healing 포함).

전체 파이프라인:

    START → design_agent → develop_agent → verify_agent ─┬─ 통과/재시도 초과 → END
                ↑                ↑                        │
                │                └── 코드 오류(develop) ──┤
                └──── 아키텍처 오류(design) ──────────────┘
"""

from __future__ import annotations

import os

from langgraph.graph import END, START, StateGraph

from ai_engine.agents.design.agent import design_node
from ai_engine.agents.develop.agent import develop_node
from ai_engine.agents.verify.agent import verify_node
from ai_engine.state.graph_state import GraphState

# Self-Healing 최대 재시도 횟수 (초과 시 마지막 결과와 리포트를 그대로 반환)
MAX_HEAL_RETRIES = int(os.environ.get("AI_MAX_HEAL_RETRIES", "2"))


def _initial_state(**overrides) -> GraphState:
    """모든 키가 채워진 초기 상태를 만든다."""
    state: GraphState = {
        "user_requirements": "",
        "rag_context": "",
        "yaml_output": "",
        "terraform_files": {},
        "validation_report": {},
        "validation_passed": False,
        "fix_target": None,
        "feedback": "",
        "retry_count": 0,
        "design_heal_count": 0,
        "messages": [],
    }
    state.update(overrides)
    return state


# ─── 라우팅 ──────────────────────────────────────────────────────────────────

def route_after_verify(state: GraphState) -> str:
    """검증 결과에 따라 Self-Healing 경로를 결정한다.

    - 통과 → END
    - 재시도 횟수 초과 → END (마지막 결과 + 검증 리포트 반환)
    - 아키텍처 문제 → design_agent 재실행
    - 코드 문제 → develop_agent 재실행
    """
    if state.get("validation_passed"):
        return "end"
    if state.get("retry_count", 0) > MAX_HEAL_RETRIES:
        print(f"[워크플로우] 재시도 {MAX_HEAL_RETRIES}회 초과 — 마지막 결과를 반환합니다.")
        return "end"
    return state.get("fix_target") or "develop"


# ─── 그래프 빌드 (모듈 로드 시 1회 컴파일) ──────────────────────────────────

def build_design_graph():
    """설계 에이전트만 포함한 그래프."""
    graph = StateGraph(GraphState)
    graph.add_node("design_agent", design_node)
    graph.add_edge(START, "design_agent")
    graph.add_edge("design_agent", END)
    return graph.compile()


def build_develop_graph():
    """개발 에이전트만 포함한 그래프."""
    graph = StateGraph(GraphState)
    graph.add_node("develop_agent", develop_node)
    graph.add_edge(START, "develop_agent")
    graph.add_edge("develop_agent", END)
    return graph.compile()


def build_verify_graph():
    """검증 에이전트만 포함한 그래프 (단독 검증용 — Self-Healing 없음)."""
    graph = StateGraph(GraphState)
    graph.add_node("verify_agent", verify_node)
    graph.add_edge(START, "verify_agent")
    graph.add_edge("verify_agent", END)
    return graph.compile()


def build_graph():
    """설계 → 개발 → 검증 (+Self-Healing 루프) 전체 파이프라인 그래프."""
    graph = StateGraph(GraphState)

    graph.add_node("design_agent", design_node)
    graph.add_node("develop_agent", develop_node)
    graph.add_node("verify_agent", verify_node)

    graph.add_edge(START, "design_agent")
    graph.add_edge("design_agent", "develop_agent")
    graph.add_edge("develop_agent", "verify_agent")
    graph.add_conditional_edges(
        "verify_agent",
        route_after_verify,
        {
            "design": "design_agent",
            "develop": "develop_agent",
            "end": END,
        },
    )

    return graph.compile()


# 요청마다 다시 컴파일하지 않도록 모듈 로드 시 1회 컴파일해 재사용
_design_app = build_design_graph()
_develop_app = build_develop_graph()
_verify_app = build_verify_graph()
_pipeline_app = build_graph()


# ─── 실행 헬퍼 ───────────────────────────────────────────────────────────────

def run_design_agent(user_requirements: str) -> dict:
    """설계 에이전트만 실행한다."""
    result = _design_app.invoke(_initial_state(user_requirements=user_requirements))
    return {
        "yaml_output": result["yaml_output"],
        "rag_context": result["rag_context"],
    }


def run_develop_agent(yaml_output: str) -> dict:
    """개발 에이전트만 실행한다. 설계 결과(yaml_output)를 직접 넘길 때 사용."""
    result = _develop_app.invoke(_initial_state(yaml_output=yaml_output))
    return {
        "terraform_files": result["terraform_files"],
    }


def run_verify_agent(yaml_output: str, terraform_files: dict) -> dict:
    """검증 에이전트만 실행한다 (Self-Healing 없이 검증 리포트만)."""
    result = _verify_app.invoke(
        _initial_state(yaml_output=yaml_output, terraform_files=terraform_files)
    )
    return {
        "validation_report": result["validation_report"],
        "validation_passed": result["validation_passed"],
    }


def run_pipeline(user_requirements: str) -> dict:
    """설계 → 개발 → 검증(Self-Healing) 전체 파이프라인을 실행한다."""
    # 재귀 한도: 노드 수(3) × (1 + 최대 재시도) + 여유
    recursion_limit = 3 * (MAX_HEAL_RETRIES + 1) + 5
    result = _pipeline_app.invoke(
        _initial_state(user_requirements=user_requirements),
        config={"recursion_limit": recursion_limit},
    )
    return {
        "yaml_output": result["yaml_output"],
        "rag_context": result["rag_context"],
        "terraform_files": result["terraform_files"],
        "validation_report": result["validation_report"],
        "validation_passed": result["validation_passed"],
    }
