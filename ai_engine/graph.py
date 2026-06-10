"""LangGraph 그래프 정의: 설계 → 개발 에이전트 워크플로우."""

from langgraph.graph import END, START, StateGraph

from ai_engine.agents.design.agent import design_node
from ai_engine.agents.develop.agent import develop_node
from ai_engine.state.graph_state import GraphState


def build_design_graph() -> StateGraph:
    """설계 에이전트만 포함한 그래프."""
    graph = StateGraph(GraphState)

    graph.add_node("design_agent", design_node)

    graph.add_edge(START, "design_agent")
    graph.add_edge("design_agent", END)

    return graph.compile()


def build_develop_graph() -> StateGraph:
    """개발 에이전트만 포함한 그래프."""
    graph = StateGraph(GraphState)

    graph.add_node("develop_agent", develop_node)

    graph.add_edge(START, "develop_agent")
    graph.add_edge("develop_agent", END)

    return graph.compile()


def build_graph() -> StateGraph:
    """설계 → 개발 전체 파이프라인 그래프."""
    graph = StateGraph(GraphState)

    graph.add_node("design_agent", design_node)
    graph.add_node("develop_agent", develop_node)

    graph.add_edge(START, "design_agent")
    graph.add_edge("design_agent", "develop_agent")
    graph.add_edge("develop_agent", END)

    return graph.compile()


def run_design_agent(user_requirements: str) -> dict:
    """설계 에이전트만 실행한다."""
    app = build_design_graph()

    initial_state: GraphState = {
        "user_requirements": user_requirements,
        "rag_context": "",
        "yaml_output": "",
        "terraform_files": {},
        "messages": [],
    }

    result = app.invoke(initial_state)
    return {
        "yaml_output": result["yaml_output"],
        "rag_context": result["rag_context"],
    }


def run_develop_agent(yaml_output: str) -> dict:
    """개발 에이전트만 실행한다. 설계 결과(yaml_output)를 직접 넘길 때 사용."""
    app = build_develop_graph()

    initial_state: GraphState = {
        "user_requirements": "",
        "rag_context": "",
        "yaml_output": yaml_output,
        "terraform_files": {},
        "messages": [],
    }

    result = app.invoke(initial_state)
    return {
        "terraform_files": result["terraform_files"],
    }


def run_pipeline(user_requirements: str) -> dict:
    """설계 → 개발 전체 파이프라인을 실행한다."""
    app = build_graph()

    initial_state: GraphState = {
        "user_requirements": user_requirements,
        "rag_context": "",
        "yaml_output": "",
        "terraform_files": {},
        "messages": [],
    }

    result = app.invoke(initial_state)
    return {
        "yaml_output": result["yaml_output"],
        "rag_context": result["rag_context"],
        "terraform_files": result["terraform_files"],
    }
