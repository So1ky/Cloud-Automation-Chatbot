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
    R7 NAT/IGW 경로 없는 서브넷의 ECS 서비스(assign_public_ip=false) / VPC Lambda에 VPC 엔드포인트 없음
       → ECR 이미지 풀·AWS API 호출 불가로 태스크가 기동조차 못 함
    R8 EventBridge 규칙 타깃/Scheduler가 ECS 태스크를 실행하는데 그 역할에 iam:PassRole 없음
    R9 CloudTrail 대상 S3 버킷 정책에 cloudtrail 주체의 s3:GetBucketAcl + s3:PutObject 없음
    R10 CloudTrail → CloudWatch Logs 역할에 logs:CreateLogStream/PutLogEvents 없음 (또는 역할 자체 누락)
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


def _subnet_refs(text: str) -> set:
    return set(re.findall(r"aws_subnet\.(\w+)", text))


def _isolated_subnets(blocks) -> set:
    """NAT/IGW 경로가 전혀 없는 라우트 테이블에 연결된 서브넷 이름 집합."""
    has_egress: Dict[str, bool] = {}
    for addr, text in blocks:
        if addr.startswith("resource.aws_route_table."):
            name = addr.split(".", 2)[2]
            has_egress[name] = bool(re.search(r"\b(nat_gateway_id|gateway_id)\s*=", text))
    for addr, text in blocks:
        if addr.startswith("resource.aws_route."):
            m = re.search(r"route_table_id\s*=\s*aws_route_table\.(\w+)", text)
            if m and re.search(r"\b(nat_gateway_id|gateway_id)\s*=", text):
                has_egress[m.group(1)] = True
    isolated = set()
    for addr, text in blocks:
        if not addr.startswith("resource.aws_route_table_association."):
            continue
        rt = re.search(r"route_table_id\s*=\s*aws_route_table\.(\w+)", text)
        if rt and has_egress.get(rt.group(1)) is False:
            isolated |= _subnet_refs(text)
    return isolated


def _vpc_endpoints(blocks) -> List[dict]:
    issues = []
    isolated = _isolated_subnets(blocks)
    if not isolated:
        return issues
    endpoint_text = "\n".join(t for a, t in blocks if a.startswith("resource.aws_vpc_endpoint."))
    has_ecr = "ecr.api" in endpoint_text and "ecr.dkr" in endpoint_text
    for addr, text in blocks:
        if addr.startswith("resource.aws_ecs_service."):
            net = re.search(r"network_configuration\s*\{([^{}]*)\}", text)
            if not net or re.search(r"assign_public_ip\s*=\s*true", net.group(1)):
                continue
            hit = _subnet_refs(net.group(1)) & isolated
            if hit and not has_ecr:
                issues.append(_issue(
                    f"{addr}가 NAT/IGW 경로 없는 서브넷({', '.join(sorted(hit))})에서 assign_public_ip=false로 "
                    f"실행되는데 ecr.api/ecr.dkr VPC 엔드포인트가 없어 컨테이너 이미지를 받지 못해 태스크가 기동하지 않습니다.",
                    "프라이빗 라우트 테이블에 S3 Gateway 엔드포인트를, 프라이빗 서브넷에 Interface 엔드포인트 "
                    "ecr.api·ecr.dkr·logs(+태스크가 호출하는 secretsmanager/kms/sqs/sns/events 등)를 "
                    "private_dns_enabled=true, VPC CIDR 443 인바운드 전용 보안그룹으로 추가하세요. "
                    "또는 NAT Gateway를 두세요(아웃바운드 전용이라 서비스가 외부에 노출되지는 않습니다).",
                    category="architecture"))
        elif addr.startswith("resource.aws_lambda_function."):
            cfg = re.search(r"vpc_config\s*\{([^{}]*)\}", text)
            if not cfg:
                continue
            hit = _subnet_refs(cfg.group(1)) & isolated
            if hit and not endpoint_text:
                issues.append(_issue(
                    f"{addr}가 NAT/IGW 경로 없는 서브넷({', '.join(sorted(hit))})에 VPC 연결되어 있는데 "
                    f"VPC 엔드포인트가 하나도 없어 DynamoDB/S3/SQS/SNS/CloudWatch Logs 등 AWS API를 호출하지 못합니다.",
                    "Lambda가 호출하는 서비스별 VPC 엔드포인트(S3·DynamoDB는 Gateway, 나머지는 Interface)를 추가하거나, "
                    "VPC 내부 리소스(RDS/ElastiCache)를 쓰지 않는다면 vpc_config를 제거하세요.",
                    category="architecture"))
    return issues


def _role_policy_text(role: str, blocks, by_addr: Dict[str, str]) -> str | None:
    """역할에 붙은 정책 본문을 모두 이어 붙인다. 내용을 알 수 없는 관리형 정책 ARN이 붙어 있으면 None."""
    texts = [by_addr.get(f"resource.aws_iam_role.{role}", "")]
    if re.search(r"managed_policy_arns\s*=", texts[0]):
        return None
    ref = rf"role\s*=\s*aws_iam_role\.{role}\.(?:id|name)\b"
    for addr, text in blocks:
        if addr.startswith("resource.aws_iam_role_policy.") and re.search(ref, text):
            texts.append(_with_policy_documents(text, by_addr))
        elif addr.startswith("resource.aws_iam_role_policy_attachment.") and re.search(ref, text):
            m = re.search(r"policy_arn\s*=\s*aws_iam_policy\.(\w+)\.arn", text)
            if not m:
                return None  # AWS 관리형 정책 등 내용을 모름 → 판정 보류
            texts.append(_with_policy_documents(by_addr.get(f"resource.aws_iam_policy.{m.group(1)}", ""), by_addr))
    texts[0] = _with_policy_documents(texts[0], by_addr)
    return "\n".join(texts)


def _has_action(policy: str, action: str) -> bool:
    service = action.split(":")[0]
    return action in policy or f"{service}:*" in policy or re.search(r'Action\s*=\s*"\*"', policy) is not None


def _ecs_pass_role(blocks) -> List[dict]:
    issues = []
    by_addr = dict(blocks)
    for addr, text in blocks:
        if addr.startswith("resource.aws_cloudwatch_event_target.") and "ecs_target" in text:
            pass
        elif addr.startswith("resource.aws_scheduler_schedule.") and "ecs_parameters" in text:
            pass
        else:
            continue
        role = re.search(r"role_arn\s*=\s*aws_iam_role\.(\w+)\.arn", text)
        task = re.search(r"task_definition_arn\s*=\s*aws_ecs_task_definition\.(\w+)\.", text)
        if not (role and task):
            continue
        task_text = by_addr.get(f"resource.aws_ecs_task_definition.{task.group(1)}", "")
        if not re.search(r"\b(execution_role_arn|task_role_arn)\s*=", task_text):
            continue
        policy = _role_policy_text(role.group(1), blocks, by_addr)
        if policy is None or _has_action(policy, "iam:PassRole"):
            continue
        issues.append(_issue(
            f"{addr}가 aws_ecs_task_definition.{task.group(1)}을 실행하는데 실행 역할 aws_iam_role.{role.group(1)}에 "
            f"iam:PassRole 권한이 없어 태스크의 execution/task 역할을 전달하지 못해 예약 실행이 매번 실패합니다.",
            f"aws_iam_role.{role.group(1)}의 정책에 ecs:RunTask와 함께 Action \"iam:PassRole\", "
            f"Resource = [태스크 정의의 execution_role_arn, task_role_arn] 문장을 추가하세요.",
            category="architecture"))
    return issues


def _cloudtrail(blocks) -> List[dict]:
    issues = []
    by_addr = dict(blocks)
    for addr, text in blocks:
        if not addr.startswith("resource.aws_cloudtrail."):
            continue
        # R9: 대상 버킷 정책
        b = re.search(r"s3_bucket_name\s*=\s*aws_s3_bucket\.(\w+)\.(?:id|bucket)\b", text)
        if b:
            bucket = b.group(1)
            ref = rf"bucket\s*=\s*aws_s3_bucket\.{bucket}\.(?:id|bucket)\b"
            policies = "\n".join(_with_policy_documents(t, by_addr) for a, t in blocks
                                  if a.startswith("resource.aws_s3_bucket_policy.") and re.search(ref, t))
            ok = ("cloudtrail.amazonaws.com" in policies and "s3:GetBucketAcl" in policies
                  and "s3:PutObject" in policies)
            if not ok:
                issues.append(_issue(
                    f"{addr}의 대상 버킷 aws_s3_bucket.{bucket}에 cloudtrail.amazonaws.com의 s3:GetBucketAcl과 "
                    f"s3:PutObject를 모두 허용하는 버킷 정책이 없어 트레일 생성이 InsufficientS3BucketPolicyException으로 실패합니다.",
                    f"aws_s3_bucket_policy(bucket = aws_s3_bucket.{bucket}.id)에 Principal {{Service = \"cloudtrail.amazonaws.com\"}} "
                    f"문장 두 개를 넣으세요: s3:GetBucketAcl on 버킷 ARN, s3:PutObject on "
                    f"\"${{버킷 ARN}}/<s3_key_prefix/>AWSLogs/${{account_id}}/*\" (Condition s3:x-amz-acl = bucket-owner-full-control). "
                    f"aws_cloudtrail에는 depends_on = [그 버킷 정책]을 두세요.",
                    category="architecture"))
            # R9b: s3_key_prefix가 있으면 CloudTrail은 <prefix>/AWSLogs/... 에 기록한다
            prefix = re.search(r's3_key_prefix\s*=\s*"([^"$]+)"', text)
            if ok and prefix:
                p = prefix.group(1).strip("/")
                if f"{p}/AWSLogs" not in policies and not re.search(r'\.arn\}/\*"', policies):
                    issues.append(_issue(
                        f"{addr}의 s3_key_prefix = \"{p}\"라 CloudTrail은 \"{p}/AWSLogs/<account_id>/\" 아래에 기록하지만 "
                        f"aws_s3_bucket.{bucket} 버킷 정책의 s3:PutObject 경로에 이 접두어가 없어 "
                        f"트레일 생성이 InsufficientS3BucketPolicyException으로 실패합니다.",
                        f"버킷 정책 PutObject Resource를 \"${{버킷 ARN}}/{p}/AWSLogs/${{account_id}}/*\"로 맞추거나 "
                        f"s3_key_prefix를 제거하세요.",
                        category="architecture"))
        # R10: CloudWatch Logs 전달 역할
        if not re.search(r"cloud_watch_logs_group_arn\s*=", text):
            continue
        role = re.search(r"cloud_watch_logs_role_arn\s*=\s*aws_iam_role\.(\w+)\.arn", text)
        if not re.search(r"cloud_watch_logs_role_arn\s*=", text):
            issues.append(_issue(
                f"{addr}에 cloud_watch_logs_group_arn만 있고 cloud_watch_logs_role_arn이 없어 CloudWatch Logs 전달이 거부됩니다.",
                "cloudtrail.amazonaws.com 신뢰 역할을 만들어 logs:CreateLogStream·logs:PutLogEvents를 로그 그룹 ARN에 허용하고 "
                "cloud_watch_logs_role_arn에 지정하세요.",
                category="architecture"))
            continue
        if not role:
            continue
        policy = _role_policy_text(role.group(1), blocks, by_addr)
        if policy is None:
            continue
        if not (_has_action(policy, "logs:CreateLogStream") and _has_action(policy, "logs:PutLogEvents")):
            issues.append(_issue(
                f"{addr}의 CloudWatch Logs 전달 역할 aws_iam_role.{role.group(1)}에 logs:CreateLogStream·logs:PutLogEvents "
                f"권한이 없어 트레일 생성 또는 로그 전달이 실패합니다.",
                f"aws_iam_role_policy(role = aws_iam_role.{role.group(1)}.id)로 Action [\"logs:CreateLogStream\", "
                f"\"logs:PutLogEvents\"], Resource = \"${{로그 그룹 ARN}}:*\" 를 추가하세요.",
                category="architecture"))
    return issues


def _sns_kms_alarms(blocks) -> List[dict]:
    """R11: KMS로 암호화된 SNS 토픽을 CloudWatch 알람이 쓰면 키 정책에 cloudwatch 주체가 필요하다."""
    issues = []
    by_addr = dict(blocks)
    alarms = "\n".join(t for a, t in blocks if a.startswith((
        "resource.aws_cloudwatch_metric_alarm.", "resource.aws_cloudwatch_composite_alarm.")))
    for addr, text in blocks:
        if not addr.startswith("resource.aws_sns_topic."):
            continue
        topic = addr.split(".", 2)[2]
        if not re.search(rf"aws_sns_topic\.{topic}\.arn\b", alarms):
            continue
        key = re.search(r"kms_master_key_id\s*=\s*(?:aws_kms_key\.(\w+)\.\w+|\"(alias/aws/sns)\")", text)
        if not key:
            continue
        if key.group(2):
            reason = "AWS 관리형 키(alias/aws/sns)는 키 정책을 바꿀 수 없어 CloudWatch가 사용할 수 없습니다"
        else:
            name = key.group(1)
            policy = "\n".join(_with_policy_documents(t, by_addr) for a, t in blocks
                                if a == f"resource.aws_kms_key.{name}"
                                or (a.startswith("resource.aws_kms_key_policy.")
                                    and re.search(rf"aws_kms_key\.{name}\.", t)))
            if "cloudwatch.amazonaws.com" in policy:
                continue
            reason = f"aws_kms_key.{name}의 키 정책에 cloudwatch.amazonaws.com 주체가 없습니다"
        issues.append(_issue(
            f"CloudWatch 알람이 {addr}로 알림을 보내지만 토픽이 KMS로 암호화되어 있고 {reason}. "
            f"알람 상태는 바뀌어도 알림은 오류 없이 전달되지 않습니다.",
            "고객 관리형 키를 쓰고 키 정책에 Principal {Service = \"cloudwatch.amazonaws.com\"}, "
            "Action [\"kms:Decrypt\", \"kms:GenerateDataKey*\"] 문장을 추가하세요 "
            "(키 정책 안에서 토픽 ARN을 참조하면 순환 참조가 되므로 조건 없이 두세요).",
            category="architecture"))
    return issues


def lint_hcl(terraform_files: Dict[str, str]) -> List[dict]:
    """리소스 내부 설정 결함 목록 (전부 critical/develop)."""
    if not terraform_files:
        return []
    blocks = _blocks(terraform_files)
    return (_rds(blocks) + _rds_engine_version(blocks) + _cloudfront(blocks)
            + _sqs_lambda(blocks) + _security_groups(blocks) + _vpc_endpoints(blocks)
            + _ecs_pass_role(blocks) + _cloudtrail(blocks) + _sns_kms_alarms(blocks))
