"""LangGraph 그래프 정의: 설계 에이전트 워크플로우."""

from langgraph.graph import END, START, StateGraph

from ai_engine.agents.design_agent import design_node
from ai_engine.state.graph_state import GraphState


def build_graph() -> StateGraph:
    """설계 에이전트 그래프를 빌드해 컴파일된 그래프를 반환한다."""
    graph = StateGraph(GraphState)

    graph.add_node("design_agent", design_node)

    graph.add_edge(START, "design_agent")
    graph.add_edge("design_agent", END)

    return graph.compile()


def run_design_agent(user_requirements: str) -> dict:
    """사용자 요구사항을 받아 설계 에이전트를 실행하고 결과를 반환한다."""
    app = build_graph()

    initial_state: GraphState = {
        "user_requirements": user_requirements,
        "rag_context": "",
        "yaml_output": "",
        "messages": [],
    }

    result = app.invoke(initial_state)
    return {
        "yaml_output": result["yaml_output"],
        "rag_context": result["rag_context"],
    }
