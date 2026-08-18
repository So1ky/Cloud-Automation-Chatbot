"""명세(YAML) 컴포넌트 ↔ Terraform 코드 리소스 대조 검사 (규칙 기반, 결정적).

개발 에이전트가 큰 아키텍처를 생성할 때 컴포넌트를 통째로 누락하는 문제를 잡는다.
LLM 산문 피드백 대신 "누락: EncodingWorker(aws_lambda_function)" 형태의
정확한 피드백을 생성해 Self-Healing 수렴을 돕는다.

보수적 원칙: 확실한 누락만 잡는다 (리소스 타입별 최소 개수 대조).
"""

from __future__ import annotations

import re
from typing import List

import yaml

# resource "aws_xxx" "name" 선언 추출
RESOURCE_RE = re.compile(r'resource\s+"([\w-]+)"\s+"([\w-]+)"')

# compute.type → 필요한 Terraform 리소스 타입 (동의어는 튜플로)
# ECS와 Fargate는 같은 리소스(aws_ecs_*)를 쓰므로 검사 시 한 그룹으로 합친다
COMPUTE_RESOURCE_MAP = {
    "EC2": [("aws_instance",)],
    "EKS": [("aws_eks_cluster",)],
    "Lambda": [("aws_lambda_function",)],
}

DATABASE_RESOURCE_MAP = {
    "RDS": ("aws_db_instance", "aws_rds_cluster"),
    "DynamoDB": ("aws_dynamodb_table",),
    "ElastiCache": ("aws_elasticache_cluster", "aws_elasticache_replication_group"),
    "OpenSearch": ("aws_opensearch_domain", "aws_elasticsearch_domain"),
}

MESSAGING_RESOURCE_MAP = {
    "SQS": ("aws_sqs_queue",),
    "SNS": ("aws_sns_topic",),
    "EventBridge": ("aws_cloudwatch_event_bus", "aws_cloudwatch_event_rule", "aws_scheduler_schedule"),
}

STREAMING_RESOURCE_MAP = {
    "Kinesis": ("aws_kinesis_stream",),
    "KinesisFirehose": ("aws_kinesis_firehose_delivery_stream",),
    "MSK": ("aws_msk_cluster",),
}

STORAGE_RESOURCE_MAP = {
    "S3": ("aws_s3_bucket",),
    "EFS": ("aws_efs_file_system",),
}


def _issue(description: str, suggestion: str) -> dict:
    return {
        "severity": "critical",
        "category": "architecture",
        "description": description,
        "suggestion": suggestion,
        "fix_target": "develop",
    }


def _count(resource_types: tuple, found: dict) -> int:
    return sum(found.get(t, 0) for t in resource_types)


def lint_code(yaml_output: str, terraform_files: dict) -> List[dict]:
    """명세 컴포넌트별 필요 리소스가 코드에 있는지 대조. 누락 목록 반환."""
    issues: List[dict] = []
    try:
        parsed = yaml.safe_load(yaml_output) or {}
    except yaml.YAMLError:
        return []  # 명세 파싱 문제는 spec_lint 담당
    arch = parsed.get("architecture") or {}
    if not arch or not terraform_files:
        return []

    code = "\n".join(terraform_files.values())
    found: dict = {}
    for rtype, _ in RESOURCE_RE.findall(code):
        found[rtype] = found.get(rtype, 0) + 1

    def check(names: list, required: tuple, label: str):
        """해당 타입 컴포넌트 개수만큼 리소스가 있는지 확인."""
        need, have = len(names), _count(required, found)
        if need > have:
            issues.append(_issue(
                f"명세의 {label} 컴포넌트 {need}개({', '.join(names)}) 대비 "
                f"Terraform의 {required[0]} 리소스가 {have}개뿐입니다. "
                f"누락 추정: {', '.join(names[have:])}",
                f"누락된 컴포넌트의 {required[0]} 리소스(및 연관 IAM/보안그룹)를 추가하세요.",
            ))

    # 1. compute (타입별 그룹핑)
    compute = [c for c in (arch.get("compute") or []) if isinstance(c, dict)]
    for ctype, requirements in COMPUTE_RESOURCE_MAP.items():
        names = [c.get("name", "?") for c in compute if c.get("type") == ctype]
        if not names:
            continue
        for required in requirements:
            check(names, required, f"compute({ctype})")

    # ECS/Fargate는 동일 리소스 타입을 쓰므로 합산 검사 (개별 검사 시 서로의 리소스로 통과되는 것 방지)
    ecs_like = [c.get("name", "?") for c in compute if c.get("type") in ("ECS", "Fargate")]
    if ecs_like:
        check(ecs_like, ("aws_ecs_task_definition",), "compute(ECS/Fargate)")
        check(ecs_like, ("aws_ecs_service",), "compute(ECS/Fargate)")

    # 2. database / messaging / streaming / storage
    for section, mapping in (
        ("database", DATABASE_RESOURCE_MAP),
        ("messaging", MESSAGING_RESOURCE_MAP),
        ("streaming", STREAMING_RESOURCE_MAP),
        ("storage", STORAGE_RESOURCE_MAP),
    ):
        items = [i for i in (arch.get(section) or []) if isinstance(i, dict)]
        for itype, required in mapping.items():
            names = [i.get("name", "?") for i in items if i.get("type") == itype]
            if names:
                check(names, required, f"{section}({itype})")

    # 3. 단일 존재형 컴포넌트
    singles = [
        (bool(arch.get("load_balancer")), ("aws_lb", "aws_alb"), "load_balancer", "ALB/NLB"),
        (bool(arch.get("cdn")), ("aws_cloudfront_distribution",), "cdn", "CloudFront"),
        (bool(arch.get("auth")), ("aws_cognito_user_pool",), "auth", "Cognito"),
        (bool(arch.get("api_gateway")), ("aws_api_gateway_rest_api", "aws_apigatewayv2_api"), "api_gateway", "API Gateway"),
        (bool(arch.get("vpc")), ("aws_vpc",), "vpc", "VPC"),
        (bool((arch.get("security") or {}).get("waf")), ("aws_wafv2_web_acl",), "security.waf", "WAF"),
        (bool((arch.get("security") or {}).get("guard_duty")), ("aws_guardduty_detector",), "security.guard_duty", "GuardDuty"),
        (bool((arch.get("security") or {}).get("kms")), ("aws_kms_key",), "security.kms", "KMS"),
    ]
    for present, required, label, human in singles:
        if present and _count(required, found) == 0:
            issues.append(_issue(
                f"명세에 {label}이(가) 정의되어 있지만 Terraform에 {required[0]} 리소스가 없습니다.",
                f"{human} 리소스를 추가하세요.",
            ))

    # 4. IAM 역할 개수
    iam = [i for i in (arch.get("iam") or []) if isinstance(i, dict)]
    if iam and found.get("aws_iam_role", 0) < len(iam):
        names = [i.get("name", "?") for i in iam]
        have = found.get("aws_iam_role", 0)
        issues.append(_issue(
            f"명세의 IAM 역할 {len(iam)}개({', '.join(names)}) 대비 aws_iam_role이 {have}개뿐입니다. "
            f"누락 추정: {', '.join(names[have:])}",
            "누락된 IAM 역할과 정책 연결을 추가하세요.",
        ))

    # 5. auto_scaling
    if arch.get("auto_scaling") and _count(
        ("aws_appautoscaling_target", "aws_autoscaling_group"), found
    ) == 0:
        issues.append(_issue(
            "명세에 auto_scaling이 정의되어 있지만 aws_appautoscaling_target/aws_autoscaling_group 리소스가 없습니다.",
            "Auto Scaling 대상 및 정책 리소스를 추가하세요.",
        ))

    return issues
