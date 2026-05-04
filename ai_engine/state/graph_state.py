from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """설계 에이전트 LangGraph 공유 상태."""

    # 사용자 자연어 요구사항
    user_requirements: str

    # RAG 검색으로 가져온 Well-Architected 문서 컨텍스트
    rag_context: str

    # 설계 에이전트가 생성한 YAML 아키텍처 명세
    yaml_output: str

    # 에이전트 간 메시지 히스토리 (LangGraph 내장 reducer)
    messages: Annotated[list, add_messages]
