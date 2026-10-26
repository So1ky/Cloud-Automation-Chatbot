"""개발 에이전트: 설계 에이전트의 YAML 명세 → Terraform HCL 코드 생성."""

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import APIConnectionError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field

from ai_engine.agents.develop.prompts import SYSTEM_PROMPT
from ai_engine.config import get_llm
from ai_engine.state.graph_state import GraphState

# resource "aws_xxx" "name" 선언 추출 (증분 수정 시 리소스 유실 감지용)
_RESOURCE_RE = re.compile(r'resource\s+"([\w-]+)"\s+"([\w-]+)"')


def _resource_set(files: dict) -> set:
    return set(_RESOURCE_RE.findall("\n".join(files.values())))


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
        print("[개발 에이전트] 검증 피드백 반영하여 코드 수정 중 (증분 수정 모드)...")
        previous_code = "\n\n".join(
            f"### {name}\n```hcl\n{content}\n```" for name, content in previous_files.items()
        )
        human_content += (
            f"\n\n---\n\n"
            f"## 기준 코드 (이전 생성본 — 검증 실패)\n\n{previous_code}\n\n"
            f"## 검증 에이전트 피드백\n\n{feedback}\n\n"
            f"## 증분 수정 지시 (반드시 준수)\n"
            f"1. 위 기준 코드를 시작점으로 삼아, 피드백에서 지적된 부분만 수정/추가하세요.\n"
            f"2. 지적되지 않은 기존 리소스는 단 하나도 삭제하거나 이름을 바꾸지 마세요.\n"
            f"3. 누락 컴포넌트 추가 시 필요한 IAM 역할/보안그룹/변수도 함께 추가하세요.\n"
            f"4. 수정 결과로 4개 파일의 전체 내용을 출력하세요 (기준 코드 + 수정사항)."
        )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    print(f"[개발 에이전트] Terraform 코드 생성 중... ({llm.model_name})")
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

    # 증분 수정 모드 보호장치: 재생성 중 기존 리소스가 유실되면 1회 복구 시도.
    # (피드백이 명시적으로 삭제를 요구한 경우는 드물고, 그 경우 다음 검증에서 다시 걸러진다)
    if is_retry and previous_files:
        dropped = _resource_set(previous_files) - _resource_set(terraform_files)
        if dropped:
            dropped_list = ", ".join(f"{t}.{n}" for t, n in sorted(dropped))
            print(f"[개발 에이전트] 유실 리소스 {len(dropped)}개 감지 → 복구 재생성: {dropped_list}")
            generated_code = "\n\n".join(
                f"### {name}\n{content}" for name, content in terraform_files.items()
            )
            recovery_messages = messages + [
                AIMessage(content=generated_code),
                HumanMessage(content=(
                    f"방금 출력에서 기준 코드에 있던 다음 리소스가 유실되었습니다: {dropped_list}\n"
                    f"이 리소스들을 기준 코드에서 그대로 되살리고, 방금 수정한 내용도 유지한 채 "
                    f"4개 파일 전체를 다시 출력하세요."
                )),
            ]
            try:
                recovered: TerraformFiles = structured_llm.invoke(recovery_messages)
                candidate = {
                    "providers.tf": recovered.providers_tf,
                    "variables.tf": recovered.variables_tf,
                    "main.tf":      recovered.main_tf,
                    "outputs.tf":   recovered.outputs_tf,
                }
                still_dropped = _resource_set(previous_files) - _resource_set(candidate)
                if len(still_dropped) < len(dropped):
                    terraform_files = candidate
                    dropped = still_dropped
            except Exception as e:
                print(f"[개발 에이전트] 복구 재생성 실패 (원본 결과 유지): {e}")
            if dropped:
                print(f"[개발 에이전트] 복구 후에도 유실 {len(dropped)}개 — 검증 단계에서 재확인됩니다")

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
