"""개발 에이전트: 설계 에이전트의 YAML 명세 → Terraform HCL 코드 생성."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field

from ai_engine.agents.develop.prompts import SYSTEM_PROMPT
from ai_engine.state.graph_state import GraphState


class TerraformFiles(BaseModel):
    """LLM structured output: 4개의 Terraform 파일 내용."""

    providers_tf: str = Field(description="Content of providers.tf — terraform block and AWS provider block")
    variables_tf: str = Field(description="Content of variables.tf — all input variable declarations")
    main_tf: str = Field(description="Content of main.tf — all AWS resource definitions")
    outputs_tf: str = Field(description="Content of outputs.tf — all output value declarations")


def develop_node(state: GraphState) -> GraphState:
    """LangGraph 노드: YAML 명세 → Terraform HCL 4개 파일 생성."""
    yaml_output = state["yaml_output"]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(TerraformFiles)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "## 아키텍처 명세 (YAML)\n\n"
                f"{yaml_output}\n\n"
                "위 명세를 기반으로 providers.tf, variables.tf, main.tf, outputs.tf 를 생성하세요."
            )
        ),
    ]

    print("[개발 에이전트] Terraform 코드 생성 중...")
    try:
        tf: TerraformFiles = structured_llm.invoke(messages)
    except RateLimitError as e:
        raise RuntimeError(f"OpenAI API 요청 한도 초과: {e}") from e
    except APITimeoutError as e:
        raise RuntimeError(f"OpenAI API 타임아웃: {e}") from e
    except APIConnectionError as e:
        raise RuntimeError(f"OpenAI API 연결 실패: {e}") from e
    except Exception as e:
        raise RuntimeError(f"LLM 호출 중 오류 발생: {e}") from e

    terraform_files = {
        "providers.tf": tf.providers_tf,
        "variables.tf": tf.variables_tf,
        "main.tf":      tf.main_tf,
        "outputs.tf":   tf.outputs_tf,
    }

    print("[개발 에이전트] Terraform 파일 4개 생성 완료")
    for filename, content in terraform_files.items():
        print(f"  - {filename}: {len(content.splitlines())}줄")

    return {
        **state,
        "terraform_files": terraform_files,
        "messages": state.get("messages", []) + [
            AIMessage(content=str(terraform_files)),
        ],
    }
