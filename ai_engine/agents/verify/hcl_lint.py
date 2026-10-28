"""Terraform 리소스 내부 설정의 결정적 검사 (LLM 없음).

code_lint가 "명세 컴포넌트가 코드에 있는가"를 보는 데 비해, 여기서는 과거 e2e에서 critical로
반복됐고 기계적으로 판정 가능한 설정 결함만 잡는다. LLM 검증은 같은 코드에서도 회차마다 다른
것을 집어 편차가 크므로, 일관성이 필요한 항목은 이쪽으로 옮긴다.

규칙:
    R1 RDS: storage_encrypted = true 누락 / publicly_accessible = true
    R2 CloudFront OAC/OAI 원본 버킷에 cloudfront 허용 버킷 정책 없음
    R3 CloudFront가 읽는 S3가 KMS 암호화인데 키 정책에 cloudfront.amazonaws.com 없음
    R4 SQS → Lambda event source mapping에서 큐 visibility_timeout < Lambda timeout
    R5 보안그룹 ingress 0.0.0.0/0 이 80/443 외 포트에 개방
    R6 RDS engine_version이 Extended Support 대상(MySQL 5.7/8.0, PostgreSQL ≤13) → vCPU당 추가 요금
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ai_engine.agents.develop.hcl_patch import parse_blocks


def _issue(description: str, suggestion: str, category: str = "security") -> dict:
    return {
        "severity": "critical",
        "category": category,
        "description": description,
        "suggestion": suggestion,
        "fix_target": "develop",
    }


def _blocks(files: Dict[str, str]) -> List[Tuple[str, str]]:
    """(주소, 블록 본문) 목록. 파일 경계를 넘지 않도록 파일별로 파싱한다."""
    out: List[Tuple[str, str]] = []
    for text in files.values():
        for b in parse_blocks(text):
            out.append((b.address, text[b.start:b.end]))
    return out


def _attr_int(block: str, key: str) -> int | None:
    m = re.search(rf"\b{key}\s*=\s*(\d+)\b", block)
    return int(m.group(1)) if m else None


def _attr_true(block: str, key: str) -> bool:
    return re.search(rf"\b{key}\s*=\s*true\b", block) is not None


def _rds(blocks) -> List[dict]:
    issues = []
    for addr, text in blocks:
        if not addr.startswith("resource.aws_db_instance."):
            continue
        if not _attr_true(text, "storage_encrypted"):
            issues.append(_issue(
                f"{addr}에 storage_encrypted = true가 없습니다 (저장 데이터 암호화 미설정).",
                f"{addr} 블록에 storage_encrypted = true 를 추가하세요."))
        if _attr_true(text, "publicly_accessible"):
            issues.append(_issue(
                f"{addr}가 publicly_accessible = true 로 인터넷에 노출됩니다.",
                f"{addr} 블록에서 publicly_accessible = false 로 바꾸세요."))
    return issues


# RDS Extended Support 대상 메이저 버전 (커뮤니티 지원 종료 → vCPU-시간당 추가 요금, Infracost "Extended support")
_EXTENDED_SUPPORT = {
    "mysql": ("5.7", "8.0"),
    "postgres": ("11", "12", "13"),
}


def _rds_engine_version(blocks) -> List[dict]:
    issues = []
    for addr, text in blocks:
        if not addr.startswith("resource.aws_db_instance."):
            continue
        engine = re.search(r'\bengine\s*=\s*"(\w+)"', text)
        version = re.search(r'\bengine_version\s*=\s*"([\d.]+)"', text)
        if not (engine and version):
            continue
        majors = _EXTENDED_SUPPORT.get(engine.group(1).lower())
        if not majors:
            continue
        v = version.group(1)
        if any(v == m or v.startswith(m + ".") for m in majors):
            issues.append(_issue(
                f"{addr}의 engine_version \"{v}\"({engine.group(1)})은 RDS Extended Support 대상이라 "
                f"vCPU-시간당 추가 요금이 붙습니다 (db.t3.micro 기준 월 약 146 USD, 인스턴스 요금의 7배).",
                f"{addr}의 engine_version을 "
                + ('"8.4"' if engine.group(1).lower() == "mysql" else '"16" 또는 "17"')
                + " 로 바꾸세요 (또는 engine_version을 생략해 현재 기본 버전 사용).",
                category="cost"))
    return issues


def _cloudfront_origin_buckets(text: str) -> set:
    return set(re.findall(r"aws_s3_bucket\.(\w+)\.(?:bucket_regional_domain_name|bucket_domain_name)", text))


def _with_policy_documents(text: str, by_addr: Dict[str, str]) -> str:
    """policy = data.aws_iam_policy_document.<name>.json 참조를 따라가 문서 본문을 덧붙인다."""
    for name in re.findall(r"data\.aws_iam_policy_document\.(\w+)\.json", text):
        text += "\n" + by_addr.get(f"data.aws_iam_policy_document.{name}", "")
    return text


def _cloudfront(blocks) -> List[dict]:
    issues = []
    dists = [(a, t) for a, t in blocks if a.startswith("resource.aws_cloudfront_distribution.")]
    if not dists:
        return issues
    by_addr = dict(blocks)
    bucket_policies = [_with_policy_documents(t, by_addr) for a, t in blocks
                       if a.startswith("resource.aws_s3_bucket_policy.")]
    kms_policy_text = "\n".join(
        _with_policy_documents(t, by_addr) for a, t in blocks
        if a.startswith("resource.aws_kms_key.") or a.startswith("resource.aws_kms_key_policy.")
    )
    kms_buckets = set()
    for a, t in blocks:
        if a.startswith("resource.aws_s3_bucket_server_side_encryption_configuration.") and "kms_master_key_id" in t:
            m = re.search(r"bucket\s*=\s*aws_s3_bucket\.(\w+)\.", t)
            if m:
                kms_buckets.add(m.group(1))
    for addr, text in dists:
        origins = _cloudfront_origin_buckets(text)
        uses_oac = "origin_access_control_id" in text or "origin_access_identity" in text
        if origins and uses_oac:
            if not any("cloudfront" in p.lower() for p in bucket_policies):
                issues.append(_issue(
                    f"{addr}가 OAC/OAI로 S3 원본({', '.join(sorted(origins))})을 읽지만 CloudFront를 허용하는 "
                    f"aws_s3_bucket_policy가 없어 모든 요청이 AccessDenied가 됩니다.",
                    "원본 버킷에 aws_s3_bucket_policy를 추가하세요: Principal {Service = \"cloudfront.amazonaws.com\"}, "
                    "Action s3:GetObject, Resource \"${bucket.arn}/*\", Condition StringEquals AWS:SourceArn = 배포 ARN.",
                    category="architecture"))
        encrypted = origins & kms_buckets
        if encrypted and "cloudfront.amazonaws.com" not in kms_policy_text:
            issues.append(_issue(
                f"{addr}의 원본 버킷({', '.join(sorted(encrypted))})이 고객 관리형 KMS 키로 암호화되어 있지만 "
                f"KMS 키 정책에 cloudfront.amazonaws.com 주체가 없어 CloudFront가 객체를 복호화하지 못합니다.",
                "aws_kms_key의 policy(또는 별도 aws_kms_key_policy)에 Principal {Service = \"cloudfront.amazonaws.com\"}, "
                "Action [\"kms:Decrypt\", \"kms:Encrypt\", \"kms:GenerateDataKey*\"] 문장을 추가하세요 "
                "(순환 참조를 피하려면 aws_kms_key_policy 리소스로 분리).",
                category="architecture"))
    return issues


def _sqs_lambda(blocks) -> List[dict]:
    issues = []
    by_addr = dict(blocks)
    for addr, text in blocks:
        if not addr.startswith("resource.aws_lambda_event_source_mapping."):
            continue
        q = re.search(r"event_source_arn\s*=\s*aws_sqs_queue\.(\w+)\.arn", text)
        f = re.search(r"function_name\s*=\s*aws_lambda_function\.(\w+)\.", text)
        if not (q and f):
            continue
        queue = by_addr.get(f"resource.aws_sqs_queue.{q.group(1)}", "")
        func = by_addr.get(f"resource.aws_lambda_function.{f.group(1)}", "")
        if not (queue and func):
            continue
        visibility = _attr_int(queue, "visibility_timeout_seconds")
        timeout = _attr_int(func, "timeout")
        if visibility is None:
            visibility = 30
        if timeout is None:
            timeout = 3
        if visibility < timeout:
            issues.append(_issue(
                f"aws_sqs_queue.{q.group(1)}의 visibility_timeout_seconds({visibility})가 소비자 "
                f"aws_lambda_function.{f.group(1)}의 timeout({timeout})보다 짧습니다. AWS는 이 매핑 생성을 거부하며, "
                f"허용되더라도 처리 중 메시지가 재전달됩니다.",
                f"aws_sqs_queue.{q.group(1)}에 visibility_timeout_seconds = {timeout * 6} 을 설정하세요 (Lambda timeout의 6배 권장).",
                category="architecture"))
    return issues


def _port_range_open(from_port: int | None, to_port: int | None, protocol: str | None) -> bool:
    if protocol in ("-1", "all"):
        return True
    if from_port is None or to_port is None:
        return False
    return not (from_port == to_port and from_port in (80, 443))


def _security_groups(blocks) -> List[dict]:
    issues = []
    for addr, text in blocks:
        rules: List[Tuple[str, str]] = []
        if addr.startswith("resource.aws_security_group_rule.") and re.search(r'type\s*=\s*"ingress"', text):
            rules.append((addr, text))
        elif addr.startswith("resource.aws_security_group."):
            for m in re.finditer(r"ingress\s*\{([^{}]*)\}", text):
                rules.append((addr, m.group(1)))
        for raddr, rule in rules:
            if "0.0.0.0/0" not in rule and "::/0" not in rule:
                continue
            proto = re.search(r'protocol\s*=\s*"?(-?\w+)"?', rule)
            if _port_range_open(_attr_int(rule, "from_port"), _attr_int(rule, "to_port"),
                                proto.group(1) if proto else None):
                issues.append(_issue(
                    f"{raddr}의 ingress가 0.0.0.0/0에 80/443 외 포트(from_port="
                    f"{_attr_int(rule, 'from_port')}, to_port={_attr_int(rule, 'to_port')})를 개방합니다.",
                    "소스를 상위 계층 보안그룹(source_security_group_id)이나 var.allowed_cidr로 제한하세요. "
                    "0.0.0.0/0은 인터넷 노출 ALB의 80/443에만 허용됩니다."))
    return issues


def lint_hcl(terraform_files: Dict[str, str]) -> List[dict]:
    """리소스 내부 설정 결함 목록 (전부 critical/develop)."""
    if not terraform_files:
        return []
    blocks = _blocks(terraform_files)
    return (_rds(blocks) + _rds_engine_version(blocks) + _cloudfront(blocks)
            + _sqs_lambda(blocks) + _security_groups(blocks))
