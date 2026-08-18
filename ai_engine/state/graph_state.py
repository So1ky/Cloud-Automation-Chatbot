from typing import Annotated, Optional
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """설계·개발·검증 에이전트 LangGraph 공유 상태."""

    # 사용자 자연어 요구사항
    user_requirements: str

    # RAG 검색으로 가져온 Well-Architected 문서 컨텍스트
    rag_context: str

    # 설계 에이전트가 생성한 YAML 아키텍처 명세
    yaml_output: str

    # 개발 에이전트가 생성한 Terraform 파일 (파일명 → HCL 내용)
    # 예: {"main.tf": "...", "variables.tf": "...", "outputs.tf": "...", "providers.tf": "..."}
    terraform_files: dict

    # ─── 검증 에이전트 (Self-Healing 루프) ───────────────────────────────
    # 검증 결과 리포트 (terraform validate / LLM 보안 분석 / Infracost)
    validation_report: dict

    # 전체 검증 통과 여부
    validation_passed: bool

    # 검증 실패 시 재생성 대상: "design"(아키텍처 문제) | "develop"(코드 문제) | None
    fix_target: Optional[str]

    # 검증 실패 내용을 설계/개발 에이전트에게 전달하는 피드백 텍스트
    feedback: str

    # Self-Healing 재시도 횟수 (무한 루프 방지)
    retry_count: int

    # 재설계 발동 횟수 (실행당 1회로 제한 — 설계 재생성이 코드 진행을 리셋시키는 것 방지)
    design_heal_count: int

    # 에이전트 간 메시지 히스토리 (LangGraph 내장 reducer)
    messages: Annotated[list, add_messages]
