"""개발 에이전트: 설계 에이전트의 YAML 명세 → Terraform HCL 코드 생성."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import APIConnectionError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field

from ai_engine.agents.develop.prompts import SYSTEM_PROMPT
from ai_engine.config import get_llm
from ai_engine.state.graph_state import GraphState


class TerraformFiles(BaseModel):
    """LLM structured output: 4개의 Terraform 파일 내용."""

    providers_tf: str = Field(description="Content of providers.tf — terraform block and AWS provider block")
    variables_tf: str = Field(description="Content of variables.tf — all input variable declarations")
    main_tf: str = Field(description="Content of main.tf — all AWS resource definitions")
    outputs_tf: str = Field(description="Content of outputs.tf — all output value declarations")


def develop_node(state: GraphState) -> dict:
    """LangGraph 노드: YAML 명세 → Terraform HCL 4개 파일 생성.

    검증 에이전트가 코드 문제로 되돌려 보낸 경우(feedback 존재)에는
    이전 Terraform 파일과 검증 오류를 함께 전달해 코드를 수정하도록 한다.
    """
    yaml_output = state["yaml_output"]
    feedback = state.get("feedback", "")
    previous_files = state.get("terraform_files", {})
    is_retry = bool(feedback) and state.get("fix_target") == "develop"

    llm = get_llm("develop")
    structured_llm = llm.with_structured_output(TerraformFiles)

    human_content = (
        "## 아키텍처 명세 (YAML)\n\n"
        f"{yaml_output}\n\n"
        "위 명세를 기반으로 providers.tf, variables.tf, main.tf, outputs.tf 를 생성하세요."
    )
    if is_retry and previous_files:
        print("[개발 에이전트] 검증 피드백 반영하여 코드 수정 중...")
        previous_code = "\n\n".join(
            f"### {name}\n```hcl\n{content}\n```" for name, content in previous_files.items()
        )
        human_content += (
            f"\n\n---\n\n"
            f"## 이전 생성 코드 (검증 실패)\n\n{previous_code}\n\n"
            f"## 검증 에이전트 피드백\n\n{feedback}\n\n"
            f"위 피드백에서 지적된 오류를 모두 수정한 전체 파일을 다시 생성하세요. "
            f"오류가 없는 부분은 이전 코드를 그대로 유지하세요."
        )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
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

    # messages에는 전체 코드 대신 요약만 남긴다 (히스토리 토큰 낭비 방지).
    summary = "Terraform 파일 생성 완료: " + ", ".join(
        f"{name}({len(content.splitlines())}줄)" for name, content in terraform_files.items()
    )
    return {
        "terraform_files": terraform_files,
        "messages": [AIMessage(content=summary)],
    }
