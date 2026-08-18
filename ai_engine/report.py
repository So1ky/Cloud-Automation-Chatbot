"""검증 리포트 → 사용자용 설명문 생성.

파이프라인이 다이어그램과 Terraform 코드를 반환할 때, 검증 결과(통과 여부, 남은 경고,
예상 비용)를 비전문가도 이해할 수 있는 한국어 설명문(마크다운)으로 함께 제공한다.

LLM 호출이 실패해도 설명문 없이 파이프라인이 죽지 않도록 템플릿 폴백을 갖는다.
"""

from __future__ import annotations

import json
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from ai_engine.config import get_llm

SUMMARY_PROMPT = """당신은 클라우드 인프라 컨설턴트입니다. AI가 자동 생성한 AWS 인프라
(아키텍처 다이어그램 + Terraform 코드)의 검증 결과를 사용자에게 설명하는 글을 씁니다.

독자는 클라우드 비전문가일 수 있습니다. 다음 원칙을 지키세요:
1. 한국어 마크다운으로 작성하고, 전체 길이는 400자~900자 정도로 간결하게.
2. 첫 문단: 전체 결과 요약 — 코드가 검증을 통과했는지, 바로 사용해도 되는 수준인지.
3. 경고가 있다면 "## 배포 전 확인이 필요한 사항" 섹션에서 각 경고를 설명:
   - 무엇이 문제인지 한 문장으로
   - 왜 치명적이지 않은지 (배포 자체는 가능함을 명확히)
   - 사용자가 무엇을 하면 되는지 구체적 행동 한 가지
4. 비용 정보가 있으면 "## 예상 비용" 섹션에 월 예상 비용과 비용의 대부분을 차지하는
   리소스 2~3개를 언급. 절감 팁이 있으면 한 줄 덧붙이기.
5. verify_attempts가 2 이상일 때만 "AI가 검증 과정에서 발견된 문제를 자동으로 수정했다"는
   취지를 한 문장으로 언급하세요. verify_attempts가 1이면 자동 수정 이야기를 아예 하지 마세요.
6. 과장하거나 겁주지 말 것. 사실에 없는 내용을 지어내지 말 것.
7. 제공된 JSON에 없는 정보는 쓰지 말 것."""


def _digest(report: dict, passed: bool) -> dict:
    """리포트에서 설명문 생성에 필요한 부분만 추린다 (토큰 절약)."""
    security = report.get("security_analysis", {}) or {}
    optimization = report.get("optimization_analysis", {}) or {}
    cost = report.get("cost_estimate", {}) or {}

    issues: List[dict] = []
    for issue in (security.get("issues") or []) + (optimization.get("issues") or []):
        issues.append({
            "severity": issue.get("severity"),
            "category": issue.get("category"),
            "description": issue.get("description"),
            "suggestion": issue.get("suggestion"),
        })

    return {
        "passed": passed,
        "accepted_with_warnings": report.get("accepted_with_warnings", False),
        "verify_attempts": report.get("attempts", 1),
        "terraform_validate_passed": (report.get("terraform_validate") or {}).get("passed"),
        "issues": issues[:8],
        "cost": {
            "skipped": cost.get("skipped", True),
            "total_monthly_cost_usd": cost.get("total_monthly_cost"),
            "top_resources": (cost.get("resources") or [])[:3],
            "finops_tips": [f.get("policy") for f in (cost.get("finops_issues") or [])[:3]],
        },
    }


def _fallback_summary(digest: dict) -> str:
    """LLM 실패 시 사용하는 결정적 템플릿."""
    lines = []
    if digest["passed"]:
        if digest["accepted_with_warnings"] or digest["issues"]:
            lines.append("생성된 인프라 코드는 문법·설계 검증을 통과했습니다. "
                         "배포는 가능하지만, 아래 확인 사항을 검토해 보시길 권합니다.")
        else:
            lines.append("생성된 인프라 코드는 모든 검증을 통과했습니다.")
    else:
        lines.append("일부 검증을 통과하지 못했습니다. 아래 사항을 확인해 주세요.")

    if digest["issues"]:
        lines.append("\n## 배포 전 확인이 필요한 사항")
        for issue in digest["issues"]:
            lines.append(f"- ({issue['severity']}) {issue['description']}")
            if issue.get("suggestion"):
                lines.append(f"  - 권장 조치: {issue['suggestion']}")

    cost = digest["cost"]
    if not cost["skipped"] and cost["total_monthly_cost_usd"]:
        try:
            total = f"${float(cost['total_monthly_cost_usd']):,.2f}"
        except (TypeError, ValueError):
            total = str(cost["total_monthly_cost_usd"])
        lines.append(f"\n## 예상 비용\n- 월 예상 비용: 약 {total} (USD)")
        for res in cost["top_resources"]:
            try:
                lines.append(f"- {res['name']}: ${float(res['monthly_cost']):,.2f}/월")
            except (TypeError, ValueError, KeyError):
                pass

    return "\n".join(lines)


def generate_user_summary(validation_report: dict, validation_passed: bool) -> str:
    """검증 리포트를 사용자용 한국어 설명문(마크다운)으로 변환한다."""
    digest = _digest(validation_report or {}, validation_passed)

    try:
        llm = get_llm("report")
        response = llm.invoke([
            SystemMessage(content=SUMMARY_PROMPT),
            HumanMessage(content=(
                "다음 검증 결과 JSON을 바탕으로 사용자에게 전달할 설명문을 작성하세요.\n\n"
                f"```json\n{json.dumps(digest, ensure_ascii=False, indent=2)}\n```"
            )),
        ])
        text = (response.content or "").strip()
        if text:
            return text
    except Exception as e:
        print(f"[리포트] 설명문 LLM 생성 실패 → 템플릿 사용: {e}")

    return _fallback_summary(digest)
