"""검증 에이전트: 생성된 Terraform 코드를 3단계로 검증하고 Self-Healing 경로를 결정한다.

검증 단계:
    0. spec lint            — 명세 자체의 설계 규칙 위반 (규칙 기반, 결정적)
    1. terraform validate   — 문법·참조 오류 (CLI, 결정적)
    2. LLM 보안/모범사례 분석 — 보안 취약점, 명세-코드 불일치, deprecated 문법
    3. LLM 최적화 분석       — 요구사항 대비 오버스펙, 더 싼 대안 아키텍처
    4. Infracost 비용 분석   — 월간 예상 비용 (미설치 시 skip)

라우팅(fix_target):
    - 설계 결함 critical(spec lint 또는 LLM fix_target=design) 존재 → "design"
      (설계 결함은 코드 수정으로 해결 불가, 재설계 시 코드도 재생성되므로 문법 오류보다 우선)
    - 그 외(문법 오류, 코드 수준 critical) → "develop"
    - 통과 → None
"""

from __future__ import annotations

import os
from typing import List, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import APIConnectionError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field

from ai_engine.agents.verify.code_lint import lint_code
from ai_engine.agents.verify.prompts import OPTIMIZATION_PROMPT, SYSTEM_PROMPT
from ai_engine.agents.verify.spec_lint import lint_spec
from ai_engine.config import get_llm
from ai_engine.agents.verify.tools import run_infracost, run_terraform_validate
from ai_engine.state.graph_state import GraphState


class ValidationIssue(BaseModel):
    """LLM 분석이 발견한 개별 이슈."""

    severity: Literal["critical", "warning", "info"]
    category: Literal["security", "architecture", "best_practice", "cost", "overspec"]
    description: str = Field(description="What is wrong and where (resource/file)")
    suggestion: str = Field(description="Concrete fix suggestion")
    fix_target: Literal["design", "develop"] = Field(
        description="'design' if the architecture spec itself is wrong, 'develop' if only the code is wrong"
    )


class SecurityAnalysis(BaseModel):
    """LLM structured output 루트 모델."""

    issues: List[ValidationIssue]


def _run_llm_analysis(yaml_output: str, terraform_files: dict) -> dict:
    """GPT-4o mini로 보안/아키텍처 일관성/모범사례 분석. 실패 시 skip 결과 반환."""
    code_blocks = "\n\n".join(
        f"### {name}\n```hcl\n{content}\n```" for name, content in terraform_files.items()
    )
    try:
        # API 키 미설정 시 생성자에서 예외가 발생하므로 생성도 try 안에서 수행
        llm = get_llm("verify")
        structured_llm = llm.with_structured_output(SecurityAnalysis)

        analysis: SecurityAnalysis = structured_llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"## 아키텍처 명세 (YAML)\n\n{yaml_output}\n\n"
                f"## 생성된 Terraform 코드\n\n{code_blocks}"
            )),
        ])
    except (RateLimitError, APITimeoutError, APIConnectionError) as e:
        return {"skipped": True, "reason": f"OpenAI API 오류: {e}", "issues": []}
    except Exception as e:
        return {"skipped": True, "reason": f"LLM 분석 실패: {e}", "issues": []}

    return {
        "skipped": False,
        "reason": None,
        "issues": [issue.model_dump() for issue in analysis.issues],
    }


def _run_optimization_analysis(user_requirements: str, yaml_output: str, terraform_files: dict) -> dict:
    """사용자 요구사항 대비 오버스펙/비용 최적화 여지를 LLM으로 분석한다.

    요구사항이 없으면(검증 단독 실행 등) 판단 근거가 없으므로 skip.
    """
    if not user_requirements.strip():
        return {"skipped": True, "reason": "사용자 요구사항이 없어 최적화 판단을 건너뜁니다.", "issues": []}

    code_blocks = "\n\n".join(
        f"### {name}\n```hcl\n{content}\n```" for name, content in terraform_files.items()
    )
    try:
        llm = get_llm("verify")
        structured_llm = llm.with_structured_output(SecurityAnalysis)

        analysis: SecurityAnalysis = structured_llm.invoke([
            SystemMessage(content=OPTIMIZATION_PROMPT),
            HumanMessage(content=(
                f"## 사용자 원본 요구사항\n\n{user_requirements}\n\n"
                f"## 설계된 아키텍처 명세 (YAML)\n\n{yaml_output}\n\n"
                f"## 생성된 Terraform 코드\n\n{code_blocks}"
            )),
        ])
    except (RateLimitError, APITimeoutError, APIConnectionError) as e:
        return {"skipped": True, "reason": f"OpenAI API 오류: {e}", "issues": []}
    except Exception as e:
        return {"skipped": True, "reason": f"최적화 분석 실패: {e}", "issues": []}

    return {
        "skipped": False,
        "reason": None,
        "issues": [issue.model_dump() for issue in analysis.issues],
    }


def _build_feedback(syntax: dict, llm_issues: List[dict], fix_target: str) -> str:
    """설계/개발 에이전트에게 되돌려 보낼 피드백 텍스트를 만든다."""
    lines: List[str] = []

    if syntax.get("errors"):
        lines.append("### Terraform validate 오류")
        for err in syntax["errors"]:
            loc = ""
            if err.get("filename"):
                loc = f" ({err['filename']}"
                loc += f":{err['line']})" if err.get("line") else ")"
            lines.append(f"- {err['summary']}{loc}")
            if err.get("detail"):
                lines.append(f"  상세: {err['detail']}")

    relevant = [i for i in llm_issues if i["severity"] == "critical" and i["fix_target"] == fix_target]
    if relevant:
        lines.append(f"### 분석 이슈 (critical, {fix_target} 대상)")
        for issue in relevant:
            lines.append(f"- [{issue['category']}] {issue['description']}")
            lines.append(f"  수정 제안: {issue['suggestion']}")

    return "\n".join(lines)


def verify_node(state: GraphState) -> dict:
    """LangGraph 노드: 3단계 검증 실행 → validation_report / fix_target / feedback 갱신."""
    yaml_output = state.get("yaml_output", "")
    terraform_files = state.get("terraform_files", {})
    user_requirements = state.get("user_requirements", "")

    # 0. 명세 자체의 설계 규칙 위반 검사 (규칙 기반, 결정적)
    print("[검증 에이전트] 아키텍처 명세 규칙 검사 중...")
    spec_issues = lint_spec(yaml_output)
    spec_criticals = [i for i in spec_issues if i["severity"] == "critical"]
    if spec_criticals:
        print(f"[검증 에이전트] 명세 설계 결함 {len(spec_criticals)}개 발견")

    # 0.5. 명세 컴포넌트 ↔ 코드 리소스 대조 (규칙 기반, 결정적)
    print("[검증 에이전트] 명세-코드 완전성 검사 중...")
    code_issues = lint_code(yaml_output, terraform_files)
    if code_issues:
        print(f"[검증 에이전트] 코드 누락 컴포넌트 {len(code_issues)}건 발견")

    # 1. terraform validate (문법 검증)
    print("[검증 에이전트] terraform validate 실행 중...")
    syntax = run_terraform_validate(terraform_files)
    if syntax["skipped"]:
        print(f"[검증 에이전트] validate 건너뜀: {syntax['reason']}")
    elif syntax["passed"]:
        print("[검증 에이전트] validate 통과")
    else:
        print(f"[검증 에이전트] validate 실패 (오류 {len(syntax['errors'])}개)")

    # 2. LLM 보안/모범사례 분석
    print("[검증 에이전트] LLM 보안/모범사례 분석 중...")
    security = _run_llm_analysis(yaml_output, terraform_files)
    critical_issues = [i for i in security["issues"] if i["severity"] == "critical"]
    if security["skipped"]:
        print(f"[검증 에이전트] LLM 분석 건너뜀: {security['reason']}")
    else:
        print(f"[검증 에이전트] LLM 분석 완료: 이슈 {len(security['issues'])}개 (critical {len(critical_issues)}개)")

    # 3. 요구사항 대비 최적화 분석 (오버스펙 / 더 싼 대안)
    print("[검증 에이전트] 요구사항 대비 최적화 분석 중...")
    optimization = _run_optimization_analysis(user_requirements, yaml_output, terraform_files)
    optimization_criticals = [i for i in optimization["issues"] if i["severity"] == "critical"]
    if optimization["skipped"]:
        print(f"[검증 에이전트] 최적화 분석 건너뜀: {optimization['reason']}")
    else:
        print(f"[검증 에이전트] 최적화 분석 완료: 이슈 {len(optimization['issues'])}개 "
              f"(critical {len(optimization_criticals)}개)")

    # 4. Infracost 비용 분석 (통과/실패 판정에는 미반영 — 리포트 용도)
    print("[검증 에이전트] Infracost 비용 분석 중...")
    cost = run_infracost(terraform_files)
    if cost["skipped"]:
        print(f"[검증 에이전트] Infracost 건너뜀: {cost['reason']}")
    else:
        try:
            monthly = f"{float(cost['total_monthly_cost']):,.2f}"
        except (TypeError, ValueError):
            monthly = str(cost["total_monthly_cost"])
        print(f"[검증 에이전트] 월간 예상 비용: {monthly} {cost['currency']} "
              f"(FinOps 정책 위반 {len(cost.get('finops_issues', []))}건)")

    # ─── 통과 판정 및 Self-Healing 라우팅 ───────────────────────────────
    # 수렴 보장 원칙:
    # - 결정적 검사(spec lint, code lint, terraform validate)는 항상 통과해야 한다.
    # - 비결정적인 LLM critical은 초반 N라운드(기본 2)까지만 블로킹한다.
    #   (LLM 리뷰는 라운드마다 새로운 지적을 만들어내므로, 무한정 블로킹하면 수렴하지 않음)
    # - 재설계(design)는 실행당 최대 1회 — 재설계는 코드를 처음부터 다시 만들기 때문.
    verify_round = state.get("retry_count", 0) + 1
    llm_blocking_rounds = int(os.environ.get("AI_LLM_CRITICAL_ROUNDS", "2"))
    design_heal_count = state.get("design_heal_count", 0)

    syntax_failed = (not syntax["skipped"]) and (not syntax["passed"])
    llm_criticals = critical_issues + optimization_criticals
    llm_blocking = verify_round <= llm_blocking_rounds

    deterministic_failed = syntax_failed or bool(spec_criticals) or bool(code_issues)
    passed = (not deterministic_failed) and not (llm_criticals and llm_blocking)
    accepted_with_warnings = passed and bool(llm_criticals)
    if accepted_with_warnings:
        print(f"[검증 에이전트] LLM critical {len(llm_criticals)}건은 "
              f"{llm_blocking_rounds}라운드 이후이므로 경고로 처리합니다")

    fix_target: Optional[str] = None
    if not passed:
        llm_design_criticals = (
            [i for i in llm_criticals if i["fix_target"] == "design"] if llm_blocking else []
        )
        if spec_criticals or (llm_design_criticals and design_heal_count == 0):
            # 설계 결함은 코드 수정으로 해결 불가 + 재설계하면 코드도 재생성되므로 우선.
            # 단, LLM 판단에 의한 재설계는 1회로 제한 (spec lint 위반은 예외 없이 재설계).
            fix_target = "design"
        else:
            fix_target = "develop"
    new_design_heal_count = design_heal_count + (1 if fix_target == "design" else 0)

    all_issues = spec_issues + code_issues + security["issues"] + optimization["issues"]
    feedback = "" if passed else _build_feedback(syntax, all_issues, fix_target)

    retry_count = state.get("retry_count", 0) + (0 if passed else 1)

    report = {
        "passed": passed,
        # LLM critical이 남았지만 블로킹 라운드를 지나 경고로 수용된 경우 True
        "accepted_with_warnings": accepted_with_warnings,
        "spec_lint": {"issues": spec_issues},
        "code_lint": {"issues": code_issues},
        "terraform_validate": syntax,
        "security_analysis": security,
        "optimization_analysis": optimization,
        "cost_estimate": cost,
        # 이번 실행을 포함한 총 검증 실행 횟수
        "attempts": verify_round,
    }

    status = "통과" if passed else f"실패 → {fix_target} 에이전트로 피드백 (시도 {retry_count}회)"
    print(f"[검증 에이전트] 최종 판정: {status}")

    return {
        "validation_report": report,
        "validation_passed": passed,
        "fix_target": fix_target,
        "feedback": feedback,
        "retry_count": retry_count,
        "design_heal_count": new_design_heal_count,
        "messages": [AIMessage(content=f"검증 결과: {status}")],
    }
