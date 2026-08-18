"""검증 에이전트 외부 도구 실행기: Terraform CLI, Infracost CLI.

두 도구 모두 미설치/미설정 환경에서는 예외 대신 skipped 결과를 반환한다
(graceful skip — 검증 파이프라인 전체가 죽지 않도록).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

# terraform init이 매번 AWS provider를 새로 받지 않도록 플러그인 캐시 사용
_PLUGIN_CACHE_DIR = Path(__file__).parent / ".terraform-plugin-cache"

INIT_TIMEOUT_SEC = 300      # 첫 실행 시 provider 다운로드 시간 고려
VALIDATE_TIMEOUT_SEC = 60
INFRACOST_TIMEOUT_SEC = 120


def _write_terraform_files(terraform_files: dict, target_dir: Path) -> None:
    """terraform_files dict({파일명: HCL 내용})를 디렉토리에 기록한다."""
    for filename, content in terraform_files.items():
        (target_dir / filename).write_text(content or "", encoding="utf-8")


def run_terraform_validate(terraform_files: dict) -> dict:
    """생성된 Terraform 파일을 임시 폴더에 쓰고 terraform init + validate 실행.

    Returns:
        {
          "skipped": bool, "reason": str|None,
          "passed": bool,
          "errors":   [{"summary", "detail", "filename", "line"}],
          "warnings": [{"summary", "detail", "filename", "line"}],
        }
    """
    terraform_bin = shutil.which("terraform")
    if terraform_bin is None:
        return {"skipped": True, "reason": "terraform CLI가 설치되어 있지 않습니다.",
                "passed": False, "errors": [], "warnings": []}
    if not terraform_files:
        return {"skipped": True, "reason": "검증할 Terraform 파일이 없습니다.",
                "passed": False, "errors": [], "warnings": []}

    _PLUGIN_CACHE_DIR.mkdir(exist_ok=True)
    env = {**os.environ, "TF_PLUGIN_CACHE_DIR": str(_PLUGIN_CACHE_DIR), "TF_IN_AUTOMATION": "1"}

    with tempfile.TemporaryDirectory(prefix="tf_validate_") as tmp:
        tmp_path = Path(tmp)
        _write_terraform_files(terraform_files, tmp_path)

        # 1. init (backend 없이 provider 플러그인만 설치)
        try:
            init_result = subprocess.run(
                [terraform_bin, "init", "-backend=false", "-input=false", "-no-color"],
                cwd=tmp, env=env, capture_output=True, text=True,
                timeout=INIT_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            return {"skipped": True, "reason": "terraform init 시간 초과 (네트워크 확인 필요)",
                    "passed": False, "errors": [], "warnings": []}

        if init_result.returncode != 0:
            # init 실패도 대부분 코드 문제(잘못된 required_providers 등)이므로 오류로 반환
            return {
                "skipped": False, "reason": None, "passed": False,
                "errors": [{
                    "summary": "terraform init 실패",
                    "detail": (init_result.stderr or init_result.stdout or "").strip()[-2000:],
                    "filename": None, "line": None,
                }],
                "warnings": [],
            }

        # 2. validate (-json으로 진단 정보 구조화)
        try:
            validate_result = subprocess.run(
                [terraform_bin, "validate", "-json", "-no-color"],
                cwd=tmp, env=env, capture_output=True, text=True,
                timeout=VALIDATE_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            return {"skipped": True, "reason": "terraform validate 시간 초과",
                    "passed": False, "errors": [], "warnings": []}

        try:
            report = json.loads(validate_result.stdout)
        except json.JSONDecodeError:
            return {
                "skipped": False, "reason": None, "passed": False,
                "errors": [{
                    "summary": "terraform validate 출력 파싱 실패",
                    "detail": (validate_result.stderr or validate_result.stdout or "").strip()[-2000:],
                    "filename": None, "line": None,
                }],
                "warnings": [],
            }

        errors, warnings = [], []
        for diag in report.get("diagnostics", []):
            rng = diag.get("range") or {}
            item = {
                "summary": diag.get("summary", ""),
                "detail": diag.get("detail", ""),
                "filename": rng.get("filename"),
                "line": (rng.get("start") or {}).get("line"),
            }
            if diag.get("severity") == "error":
                errors.append(item)
            else:
                warnings.append(item)

        return {
            "skipped": False,
            "reason": None,
            "passed": bool(report.get("valid", False)),
            "errors": errors,
            "warnings": warnings,
        }


def _infracost_skip(reason: str) -> dict:
    return {"skipped": True, "reason": reason, "total_monthly_cost": None, "currency": "USD",
            "potential_yearly_savings": None, "resources": [], "finops_issues": []}


def run_infracost(terraform_files: dict) -> dict:
    """Infracost(v2)로 월간 예상 비용과 FinOps 정책 위반을 분석한다. 미설치/미설정 시 skip.

    infracost v2는 `breakdown` 대신 `scan`(총비용+FinOps) / `inspect`(리소스별 비용)를 사용한다.

    Returns:
        {
          "skipped": bool, "reason": str|None,
          "total_monthly_cost": str|None, "currency": str,
          "potential_yearly_savings": str|None,
          "resources": [{"name", "monthly_cost"}],          # 비용 내림차순
          "finops_issues": [{"policy", "message", "resources"}],  # 실패한 정책만
        }
    """
    infracost_bin = shutil.which("infracost")
    if infracost_bin is None:
        return _infracost_skip("infracost CLI가 설치되어 있지 않습니다. (brew install infracost)")
    if not os.environ.get("INFRACOST_API_KEY") and not _infracost_configured():
        return _infracost_skip("infracost 인증이 없습니다. (infracost auth login)")
    if not terraform_files:
        return _infracost_skip("분석할 Terraform 파일이 없습니다.")

    with tempfile.TemporaryDirectory(prefix="tf_infracost_") as tmp:
        tmp_path = Path(tmp)
        _write_terraform_files(terraform_files, tmp_path)

        # 1. scan: 총 월비용 + FinOps 정책 검사
        try:
            scan = subprocess.run(
                [infracost_bin, "scan", "--json"],
                cwd=tmp, capture_output=True, text=True, timeout=INFRACOST_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            return _infracost_skip("infracost scan 실행 시간 초과")

        # scan은 정책 위반/진단이 있으면 exit code가 0이 아닐 수 있으므로
        # exit code와 무관하게 stdout이 유효한 JSON이면 결과를 사용한다.
        try:
            data = json.loads(scan.stdout)
        except json.JSONDecodeError:
            return _infracost_skip(
                f"infracost scan 실패: {(scan.stderr or scan.stdout or '').strip()[-500:]}"
            )

        summary = data.get("summary") or {}

        finops_issues = []
        for project in data.get("projects", []):
            for policy in project.get("finops_results") or []:
                failing = policy.get("failing_resources") or []
                if failing:
                    finops_issues.append({
                        "policy": policy.get("policy_name"),
                        "message": policy.get("policy_message"),
                        "resources": [r.get("name") if isinstance(r, dict) else str(r) for r in failing][:5],
                    })

        # 2. inspect: 리소스별 월비용 (실패해도 총비용은 반환)
        resources = []
        try:
            inspect = subprocess.run(
                [infracost_bin, "inspect", "--group-by", "resource", "--json"],
                cwd=tmp, capture_output=True, text=True, timeout=INFRACOST_TIMEOUT_SEC,
            )
            if inspect.returncode == 0:
                for row in json.loads(inspect.stdout):
                    cost = row.get("cost")
                    if cost and float(cost) > 0:
                        resources.append({
                            "name": (row.get("columns") or {}).get("resource"),
                            "monthly_cost": cost,
                        })
                resources.sort(key=lambda r: float(r["monthly_cost"]), reverse=True)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
            pass

        return {
            "skipped": False,
            "reason": None,
            "total_monthly_cost": summary.get("total_monthly_cost"),
            "currency": data.get("currency", "USD"),
            "potential_yearly_savings": summary.get("total_potential_yearly_savings"),
            "resources": resources,
            "finops_issues": finops_issues,
        }


def _infracost_configured() -> Optional[str]:
    """infracost auth login으로 저장된 인증 파일이 있는지 확인한다.

    버전/OS에 따라 저장 위치가 다르다:
    - v2 (macOS): ~/Library/Application Support/infracost/token.json
    - v1 계열:    ~/.config/infracost/credentials.yml
    """
    candidates = [
        Path.home() / "Library" / "Application Support" / "infracost" / "token.json",
        Path.home() / "Library" / "Application Support" / "infracost" / "credentials.yml",
        Path.home() / ".config" / "infracost" / "credentials.yml",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None
