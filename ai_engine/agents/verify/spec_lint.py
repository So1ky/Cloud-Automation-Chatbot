"""아키텍처 명세(YAML) 자체의 논리적 결함을 규칙 기반으로 검사한다.

LLM 분석과 달리 결정적(deterministic)이므로, 설계 에이전트로 되돌려야 하는
명세 결함(fix_target="design")을 안정적으로 잡아낸다.
검사 규칙은 설계 에이전트 시스템 프롬프트의 설계 규칙과 1:1로 대응한다.
"""

from __future__ import annotations

from typing import List

import yaml

VPC_COMPUTE_TYPES = {"EC2", "ECS", "EKS", "Fargate"}
PRIVATE_ONLY_COMPUTE = {"ECS", "EKS", "Fargate"}
VPC_DB_TYPES = {"RDS", "ElastiCache", "OpenSearch"}


def _issue(description: str, suggestion: str, severity: str = "critical") -> dict:
    return {
        "severity": severity,
        "category": "architecture",
        "description": description,
        "suggestion": suggestion,
        "fix_target": "design",
    }


def lint_spec(yaml_output: str) -> List[dict]:
    """명세 YAML의 설계 규칙 위반 목록을 반환한다. 문제 없으면 빈 리스트."""
    issues: List[dict] = []

    try:
        parsed = yaml.safe_load(yaml_output) or {}
    except yaml.YAMLError as e:
        return [_issue(f"아키텍처 명세 YAML 파싱 실패: {e}", "유효한 YAML 명세를 다시 생성하세요.")]

    arch = parsed.get("architecture") or {}
    vpc = arch.get("vpc") or {}
    subnets = [s for s in (vpc.get("subnets") or []) if isinstance(s, dict)]
    subnet_type = {s.get("name"): s.get("type") for s in subnets}
    subnet_az = {s.get("name"): s.get("az") for s in subnets}
    compute = [c for c in (arch.get("compute") or []) if isinstance(c, dict)]

    # 규칙 1: RDS/ElastiCache/OpenSearch 등 DB는 프라이빗 서브넷에만
    for db in arch.get("database") or []:
        if not isinstance(db, dict) or db.get("type") == "DynamoDB":
            continue
        public_sns = [sn for sn in (db.get("subnets") or []) if subnet_type.get(sn) == "public"]
        if public_sns:
            issues.append(_issue(
                f"데이터베이스 '{db.get('name')}'({db.get('type')})가 퍼블릭 서브넷({', '.join(public_sns)})에 배치되어 "
                f"인터넷에 노출될 수 있습니다.",
                "데이터베이스는 반드시 프라이빗 서브넷에만 배치하세요.",
            ))

    # 규칙 2: ECS/EKS/Fargate는 프라이빗 서브넷에만
    for c in compute:
        if c.get("type") in PRIVATE_ONLY_COMPUTE:
            public_sns = [sn for sn in (c.get("subnets") or []) if subnet_type.get(sn) == "public"]
            if public_sns:
                issues.append(_issue(
                    f"컴퓨팅 '{c.get('name')}'({c.get('type')})가 퍼블릭 서브넷({', '.join(public_sns)})에 배치되었습니다.",
                    "ECS/EKS/Fargate는 프라이빗 서브넷에 배치하고 ALB를 통해 트래픽을 받으세요.",
                ))

    # 규칙 3: ALB는 서로 다른 AZ의 서브넷 2개 이상 —
    # 인터넷 연결형(internal=false)은 퍼블릭, 사내 전용(internal=true)은 프라이빗 서브넷
    for lb in arch.get("load_balancer") or []:
        if not isinstance(lb, dict):
            continue
        sns = lb.get("subnets") or []
        name = lb.get("name")
        is_internal = lb.get("internal", False)
        if len(sns) < 2:
            issues.append(_issue(
                f"로드밸런서 '{name}'의 서브넷이 {len(sns)}개뿐입니다. ALB는 서로 다른 AZ의 서브넷 2개 이상이 필수입니다.",
                "서로 다른 AZ의 서브넷을 2개 이상 지정하세요.",
            ))
        else:
            azs = {subnet_az.get(sn) for sn in sns if subnet_az.get(sn)}
            if len(azs) < 2:
                issues.append(_issue(
                    f"로드밸런서 '{name}'의 서브넷들이 같은 AZ에 있습니다. ALB는 다중 AZ가 필수입니다.",
                    "서로 다른 AZ에 있는 서브넷들을 지정하세요.",
                ))
            if is_internal:
                public_sns = [sn for sn in sns if subnet_type.get(sn) == "public"]
                if public_sns:
                    issues.append(_issue(
                        f"사내 전용(internal) 로드밸런서 '{name}'가 퍼블릭 서브넷({', '.join(public_sns)})에 배치되었습니다.",
                        "internal 로드밸런서는 프라이빗 서브넷에 배치하세요.",
                    ))
            else:
                private_sns = [sn for sn in sns if subnet_type.get(sn) == "private"]
                if private_sns:
                    issues.append(_issue(
                        f"인터넷 연결 로드밸런서 '{name}'가 프라이빗 서브넷({', '.join(private_sns)})에 배치되었습니다.",
                        "인터넷 연결 ALB는 퍼블릭 서브넷에 배치하거나, 사내 전용이면 internal: true로 설정하세요.",
                    ))

    # 규칙 4: NAT Gateway는 퍼블릭 서브넷에만
    networking = arch.get("networking") or {}
    for nat in networking.get("nat_gateway") or []:
        if isinstance(nat, dict) and subnet_type.get(nat.get("subnet")) == "private":
            issues.append(_issue(
                f"NAT Gateway가 프라이빗 서브넷('{nat.get('subnet')}')에 배치되었습니다.",
                "NAT Gateway는 Internet Gateway에 도달해야 하므로 퍼블릭 서브넷에 배치하세요.",
            ))

    # 규칙 5: VPC가 필요한 리소스가 있는데 VPC/서브넷이 없음
    needs_vpc = any(c.get("type") in VPC_COMPUTE_TYPES for c in compute) or any(
        isinstance(db, dict) and db.get("type") in VPC_DB_TYPES for db in arch.get("database") or []
    )
    if needs_vpc and not subnets:
        issues.append(_issue(
            "EC2/ECS/EKS/Fargate 또는 RDS가 포함되어 있는데 VPC/서브넷 정의가 없습니다.",
            "VPC와 퍼블릭/프라이빗 서브넷을 설계에 추가하세요.",
        ))

    # 규칙 6: 순수 서버리스(관리형 서비스만)인데 VPC가 존재
    if not needs_vpc and subnets and compute:
        issues.append(_issue(
            "Lambda와 관리형 서비스만 사용하는 서버리스 아키텍처인데 불필요한 VPC가 포함되어 있습니다.",
            "VPC 없이 설계하세요. Lambda는 VPC 없이도 S3/DynamoDB/Kinesis/SQS에 접근할 수 있습니다.",
            severity="warning",
        ))

    # 규칙 7: auto_scaling.target은 compute 이름과 일치해야 함
    compute_names = {c.get("name") for c in compute}
    for asg in arch.get("auto_scaling") or []:
        if isinstance(asg, dict) and asg.get("target") not in compute_names:
            issues.append(_issue(
                f"auto_scaling target '{asg.get('target')}'과 일치하는 compute 리소스가 없습니다. "
                f"(compute: {', '.join(n for n in compute_names if n)})",
                "auto_scaling.target을 compute 리소스의 name과 정확히 일치시키세요.",
            ))

    return issues
