"""개발 에이전트: 설계 에이전트의 YAML 명세 → Terraform HCL 코드 생성."""

from typing import List, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import APIConnectionError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field

from ai_engine.agents.develop.hcl_patch import apply_patch
from ai_engine.agents.develop.prompts import SYSTEM_PROMPT
from ai_engine.config import get_llm
from ai_engine.state.graph_state import GraphState


class HclEdit(BaseModel):
    """재시도 시 한 블록 단위의 편집."""

    file: Literal["providers.tf", "variables.tf", "main.tf", "outputs.tf"]
    action: Literal["replace", "add", "delete"]
    address: str = Field(description=(
        "Top-level block address: resource.<type>.<name> (e.g. resource.aws_lb.api), "
        "data.<type>.<name>, variable.<name>, output.<name>, locals, provider.<name>, terraform"
    ))
    hcl: str = Field(default="", description=(
        "The COMPLETE new text of that single block for replace/add (from the block header to its "
        "closing brace). Empty for delete."
    ))


class TerraformPatch(BaseModel):
    """LLM structured output (재시도 모드): 전체 파일 대신 편집 목록."""

    edits: List[HclEdit] = Field(description="Only the blocks that must change to resolve the feedback.")


class TerraformFiles(BaseModel):
    """LLM structured output: 4개의 Terraform 파일 내용."""

    providers_tf: str = Field(description="Content of providers.tf — terraform block and AWS provider block")
    variables_tf: str = Field(description="Content of variables.tf — all input variable declarations")
    main_tf: str = Field(description="Content of main.tf — all AWS resource definitions")
    outputs_tf: str = Field(description="Content of outputs.tf — all output value declarations")


def _patch_previous_files(llm, yaml_output: str, previous_files: dict, feedback: str) -> dict | None:
    """재시도 모드: 검증 피드백을 편집 목록으로 받아 이전 파일에 기계적으로 적용한다.

    파일 전체를 다시 출력시키면 리소스 누락·이름 변경이 생기므로, 바뀌는 블록만 받고 나머지는
    바이트 그대로 보존한다. 적용된 편집이 하나도 없으면 None (호출 측이 전체 재생성으로 폴백).
    """
    print("[개발 에이전트] 검증 피드백 반영하여 코드 수정 중 (블록 패치 모드)...")
    previous_code = "\n\n".join(
        f"### {name}\n```hcl\n{content}\n```" for name, content in previous_files.items()
    )
    human_content = (
        "## 아키텍처 명세 (YAML)\n\n"
        f"{yaml_output}\n\n---\n\n"
        f"## 기준 코드 (검증 실패)\n\n{previous_code}\n\n"
        f"## 검증 에이전트 피드백\n\n{feedback}\n\n"
        "## 수정 지시 (반드시 준수)\n"
        "1. 파일 전체를 다시 쓰지 말고, 피드백을 해결하는 데 필요한 블록만 편집 목록(edits)으로 내세요.\n"
        "2. replace/add의 hcl에는 그 블록 하나의 전문(헤더부터 닫는 중괄호까지)을 넣고, "
        "address는 hcl의 헤더와 정확히 일치해야 합니다 (예: resource.aws_lb.api).\n"
        "3. 기존 블록의 이름을 바꾸지 마세요. 꼭 없애야 하면 delete를 쓰세요.\n"
        "4. 새 리소스가 변수·출력·IAM·보안그룹을 필요로 하면 그 블록들도 add로 함께 내세요.\n"
        "5. 지적되지 않은 블록은 건드리지 마세요."
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human_content)]
    print(f"[개발 에이전트] 편집 목록 생성 중... ({llm.model_name})")
    try:
        patch: TerraformPatch = llm.with_structured_output(TerraformPatch).invoke(messages)
    except Exception as e:
        print(f"[개발 에이전트] 편집 목록 생성 실패: {e}")
        return None
    files, applied, skipped = apply_patch(previous_files, [e.model_dump() for e in patch.edits])
    print(f"[개발 에이전트] 편집 {len(patch.edits)}건 중 적용 {len(applied)}건, 건너뜀 {len(skipped)}건")
    for a in applied:
        print(f"  + {a}")
    for r in skipped:
        print(f"  ! {r}")
    return files if applied else None


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

    if is_retry and previous_files:
        patched = _patch_previous_files(llm, yaml_output, previous_files, feedback)
        if patched is not None:
            return _finish(patched)
        print("[개발 에이전트] 패치가 하나도 적용되지 않아 전체 재생성으로 폴백합니다")

    structured_llm = llm.with_structured_output(TerraformFiles)
    human_content = (
        "## 아키텍처 명세 (YAML)\n\n"
        f"{yaml_output}\n\n"
        "위 명세를 기반으로 providers.tf, variables.tf, main.tf, outputs.tf 를 생성하세요."
    )
    if is_retry and previous_files:
        previous_code = "\n\n".join(
            f"### {name}\n```hcl\n{content}\n```" for name, content in previous_files.items()
        )
        human_content += (
            f"\n\n---\n\n"
            f"## 기준 코드 (이전 생성본 — 검증 실패)\n\n{previous_code}\n\n"
            f"## 검증 에이전트 피드백\n\n{feedback}\n\n"
            f"## 수정 지시 (반드시 준수)\n"
            f"1. 위 기준 코드를 시작점으로 삼아, 피드백에서 지적된 부분만 수정/추가하세요.\n"
            f"2. 지적되지 않은 기존 리소스는 단 하나도 삭제하거나 이름을 바꾸지 마세요.\n"
            f"3. 수정 결과로 4개 파일의 전체 내용을 출력하세요 (기준 코드 + 수정사항)."
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

    return _finish({
        "providers.tf": tf.providers_tf,
        "variables.tf": tf.variables_tf,
        "main.tf":      tf.main_tf,
        "outputs.tf":   tf.outputs_tf,
    })


def _finish(terraform_files: dict) -> dict:
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
