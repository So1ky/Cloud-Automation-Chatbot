"""우리 YAML 명세 → awslabs/diagram-as-code (awsdac) YAML 변환기.

awsdac: https://github.com/awslabs/diagram-as-code
입력: architecture_schema.py 기반 YAML 딕셔너리
출력: awsdac CLI가 바로 사용할 수 있는 YAML 딕셔너리
"""

DEFINITION_URL = (
    "https://raw.githubusercontent.com/awslabs/diagram-as-code"
    "/main/definitions/definition-for-aws-icons-light.yaml"
)

# ─── 리소스 타입 매핑 ─────────────────────────────────────────────────────────

COMPUTE_TYPE_MAP = {
    "EC2":     "AWS::EC2::Instance",
    "ECS":     "AWS::ECS::Cluster",
    "EKS":     "AWS::EKS::Cluster",
    "Lambda":  "AWS::Lambda::Function",
    "Fargate": "AWS::ECS::Cluster",
}

LB_PRESET_MAP = {
    "ALB": "Application Load Balancer",
    "NLB": "Network Load Balancer",
}

DATABASE_TYPE_MAP = {
    "RDS":         "AWS::RDS::DBInstance",
    "DynamoDB":    "AWS::DynamoDB",
    "ElastiCache": "AWS::ElastiCache::CacheCluster",
    "OpenSearch":  "AWS::OpenSearchService::Domain",
}

STORAGE_TYPE_MAP = {
    "S3":  "AWS::S3::Bucket",
    "EFS": "AWS::EFS::FileSystem",
}

MESSAGING_TYPE_MAP = {
    "SQS":         "AWS::SQS::Queue",
    "SNS":         "AWS::SNS::Topic",
    "EventBridge": "AWS::EventBridge::EventBus",
}

STREAMING_TYPE_MAP = {
    "Kinesis":         "AWS::Kinesis::Stream",
    "KinesisFirehose": "AWS::KinesisFirehose::DeliveryStream",
    "MSK":             "AWS::MSK::Cluster",
}


# ─── 변환 메인 함수 ───────────────────────────────────────────────────────────

def convert_to_diagram_yaml(arch: dict) -> dict:
    """architecture 딕셔너리를 awsdac YAML 딕셔너리로 변환한다."""
    resources: dict = {}
    links: list = []
    # 리소스를 데이터 흐름 순서대로 담을 버킷
    cloud_streaming:     list = []   # Kinesis, SQS 등 입력 큐
    cloud_compute:       list = []   # Lambda 등 처리
    cloud_notification:  list = []   # SNS 등 완료 알림 (Lambda 처리 후)
    cloud_db:            list = []   # DynamoDB 등 출력 DB
    cloud_storage:       list = []   # S3 등 출력 Storage
    cloud_entry:         list = []   # CloudFront, Route53 등 사용자 진입점 (VPC 위에 배치)
    cloud_support:       list = []   # ECR, Cognito, KMS 등 지원 서비스 (VPC 아래에 배치)

    # 서브넷별 자식 리소스 추적 (name → [resource_id, ...])
    subnet_children: dict = {
        s["name"]: [] for s in arch.get("vpc", {}).get("subnets", [])
    }
    subnet_type_map: dict = {
        s["name"]: s["type"] for s in arch.get("vpc", {}).get("subnets", [])
    }

    # ── 아키텍처 타입 사전 판별 (하위 섹션에서 공통 사용) ────────────────────
    has_cdn_local = bool(arch.get("cdn"))
    is_cdn_vpc    = has_cdn_local and bool(arch.get("vpc"))

    # ── 서버리스 아키텍처 자동 감지 ──────────────────────────────────────────
    # ECS/EKS/EC2/Fargate 없이 Lambda만 있고, Kinesis/SQS/DynamoDB/S3 등 managed 서비스만 사용하면
    # LLM이 실수로 VPC/서브넷을 넣어도 강제로 무시하고 서버리스 모드로 처리
    compute_types = {item.get("type") for item in arch.get("compute", [])}
    db_types = {item.get("type") for item in arch.get("database", [])}
    vpc_compute_types = {"ECS", "EKS", "EC2", "Fargate"}
    vpc_db_types = {"RDS"}
    is_forced_serverless = (
        bool(compute_types) and
        not compute_types.intersection(vpc_compute_types) and
        not db_types.intersection(vpc_db_types) and
        (bool(arch.get("streaming")) or bool(arch.get("messaging")))
    )
    # 서버리스로 강제 판정 시 VPC/서브넷 설정을 비움
    if is_forced_serverless:
        subnet_children = {}
        subnet_type_map = {}

    # auto_scaling 타겟 맵 구성 (compute name → auto_scaling 설정)
    auto_scaling_map: dict = {}
    for as_cfg in (arch.get("auto_scaling") or []):
        if isinstance(as_cfg, dict) and as_cfg.get("target"):
            auto_scaling_map[as_cfg["target"]] = as_cfg

    # ── 1. Compute ──────────────────────────────────────────────────────────
    compute_names: list[str] = []
    for item in arch.get("compute", []):
        rid = item["name"]
        compute_names.append(rid)
        comp_type = item.get("type", "ECS")
        resources[rid] = {"Type": COMPUTE_TYPE_MAP.get(comp_type, "AWS::EC2::Instance")}

        subnets = item.get("subnets") or []

        # Auto Scaling Group으로 감싸기
        if rid in auto_scaling_map:
            asg_rid = f"{rid}ASG"
            resources[asg_rid] = {
                "Type": "AWS::AutoScaling::AutoScalingGroup",
                "Children": [rid],
            }
            # ECS/EKS/Fargate는 반드시 Private Subnet에 배치 (LLM이 Public Subnet 지정해도 강제 이동)
            effective_subnets = subnets
            if comp_type in {"ECS", "EKS", "Fargate"} and subnets and subnet_type_map.get(subnets[0]) == "public":
                private_sn = next((n for n, t in subnet_type_map.items() if t == "private"), None)
                if private_sn:
                    effective_subnets = [private_sn] + subnets[1:]
            if effective_subnets and effective_subnets[0] in subnet_children:
                subnet_children[effective_subnets[0]].append(asg_rid)
            else:
                cloud_compute.append(asg_rid)
            # Multi-AZ: 두 번째 서브넷에 ASG 복제
            if len(subnets) >= 2 and subnets[1] in subnet_children:
                replica_rid = f"{rid}Replica"
                asg_replica_rid = f"{rid}ReplicaASG"
                resources[replica_rid] = resources[rid].copy()
                resources[asg_replica_rid] = {
                    "Type": "AWS::AutoScaling::AutoScalingGroup",
                    "Children": [replica_rid],
                }
                subnet_children[subnets[1]].append(asg_replica_rid)
                compute_names.append(replica_rid)
        else:
            # ECS/EKS/Fargate는 반드시 Private Subnet에 배치
            effective_subnets = subnets
            if comp_type in {"ECS", "EKS", "Fargate"} and subnets and subnet_type_map.get(subnets[0]) == "public":
                private_sn = next((n for n, t in subnet_type_map.items() if t == "private"), None)
                if private_sn:
                    effective_subnets = [private_sn] + subnets[1:]
            if effective_subnets and effective_subnets[0] in subnet_children:
                subnet_children[effective_subnets[0]].append(rid)
            else:
                cloud_compute.append(rid)
            # Multi-AZ: 두 번째 서브넷에 컴퓨팅 복제
            if len(effective_subnets) >= 2 and effective_subnets[1] in subnet_children:
                replica_rid = f"{rid}Replica"
                resources[replica_rid] = resources[rid].copy()
                subnet_children[effective_subnets[1]].append(replica_rid)
                compute_names.append(replica_rid)

    # ── 2. Load Balancer ────────────────────────────────────────────────────
    lb_names: list[str] = []
    for item in arch.get("load_balancer", []):
        rid = item["name"]
        lb_names.append(rid)
        lb_type = item.get("type", "ALB")
        res: dict = {"Type": "AWS::ElasticLoadBalancingV2::LoadBalancer"}
        if lb_type in LB_PRESET_MAP:
            res["Preset"] = LB_PRESET_MAP[lb_type]
        resources[rid] = res
        # ALB도 첫 번째 퍼블릭 서브넷에만 배치
        subnets = item.get("subnets", [])
        if subnets and subnets[0] in subnet_children:
            subnet_children[subnets[0]].append(rid)

    # ── 3. Database ─────────────────────────────────────────────────────────
    db_names: list[str] = []
    for item in arch.get("database", []):
        rid = item["name"]
        db_names.append(rid)
        resources[rid] = {"Type": DATABASE_TYPE_MAP.get(item["type"], "AWS::RDS::DBInstance")}
        subnets = item.get("subnets") or []
        if subnets and subnets[0] in subnet_children:
            subnet_children[subnets[0]].append(rid)
        else:
            cloud_db.append(rid)
        # Multi-AZ: 두 번째 서브넷에 DB Standby 복제
        if len(subnets) >= 2 and subnets[1] in subnet_children:
            replica_rid = f"{rid}Standby"
            resources[replica_rid] = resources[rid].copy()
            subnet_children[subnets[1]].append(replica_rid)

    # ── 4. Cache ────────────────────────────────────────────────────────────
    for item in arch.get("cache", []):
        rid = item.get("name", "ElastiCache")
        resources[rid] = {"Type": "AWS::ElastiCache::CacheCluster"}
        sn = item.get("subnet", "")
        if sn in subnet_children:
            subnet_children[sn].append(rid)

    # ── 5. API Gateway ──────────────────────────────────────────────────────
    apigw_names: list[str] = []
    for item in arch.get("api_gateway", []):
        rid = item["name"]
        apigw_names.append(rid)
        resources[rid] = {"Type": "AWS::ApiGateway::RestApi"}
        cloud_entry.append(rid)

    # ── 6. Container Registry ───────────────────────────────────────────────
    for item in arch.get("container_registry", []):
        rid = item["name"]
        resources[rid] = {"Type": "AWS::ECR::Repository"}
        cloud_support.append(rid)

    # ── 7. NAT Gateway & Internet Gateway ──────────────────────────────────
    networking = arch.get("networking") or {}
    igw_exists = networking.get("internet_gateway", False)
    nat_names: list[str] = []

    for i, nat in enumerate(networking.get("nat_gateway") or []):
        rid = f"NATGateway{i + 1}" if i > 0 else "NATGateway"
        nat_names.append(rid)
        resources[rid] = {"Type": "AWS::EC2::NatGateway"}
        sn = nat.get("subnet", "")
        # NAT Gateway는 반드시 Public Subnet에 배치 - LLM이 private subnet을 지정해도 강제 이동
        if sn not in subnet_children or subnet_type_map.get(sn) != "public":
            sn = next((n for n, t in subnet_type_map.items() if t == "public"), sn)
        if sn in subnet_children:
            subnet_children[sn].append(rid)

    # ── 8. 서브넷 → VPC 구성 ────────────────────────────────────────────────
    # 실제 리소스가 있는 서브넷만 포함 (빈 서브넷은 VPC가 비어있으면 제외)
    any_subnet_has_resource_check = any(
        len(children) > 0 for children in subnet_children.values()
    )
    public_subnets  = [n for n, t in subnet_type_map.items() if t == "public"]
    private_subnets = [n for n, t in subnet_type_map.items() if t == "private"]

    if any_subnet_has_resource_check:
        for sn_name in subnet_type_map:
            children = subnet_children.get(sn_name, [])
            sn_res: dict = {
                "Type": "AWS::EC2::Subnet",
                "Preset": "PublicSubnet" if subnet_type_map[sn_name] == "public" else "PrivateSubnet",
            }
            if children:
                sn_res["Children"] = children
            resources[sn_name] = sn_res

    # 서브넷 그룹 → HorizontalStack
    vpc_children: list = []

    if len(public_subnets) > 1:
        resources["PublicSubnetStack"] = {
            "Type": "AWS::Diagram::HorizontalStack",
            "Children": public_subnets,
        }
        vpc_children.append("PublicSubnetStack")
    elif public_subnets:
        vpc_children.extend(public_subnets)

    if len(private_subnets) > 1:
        resources["PrivateSubnetStack"] = {
            "Type": "AWS::Diagram::HorizontalStack",
            "Children": private_subnets,
        }
        vpc_children.append("PrivateSubnetStack")
    elif private_subnets:
        vpc_children.extend(private_subnets)

    # VPC는 실제 서브넷에 리소스가 하나라도 있을 때만 생성
    # 서브넷이 정의됐어도 안에 아무것도 없으면 (Lambda serverless 등) VPC 생성 안 함
    any_subnet_has_resource = any(
        len(children) > 0 for children in subnet_children.values()
    )
    has_vpc = bool(subnet_type_map) and any_subnet_has_resource
    if has_vpc:
        vpc_res: dict = {
            "Type": "AWS::EC2::VPC",
            "Direction": "vertical",  # Public Subnet(상단) → Private Subnet(하단) 수직 흐름
            "Align": "center",
            "Children": vpc_children,
        }
        # CDN이 있으면 IGW를 BorderChild로 두지 않음
        # (CloudFront → ALB 수직선이 VPC 전체를 관통하는 시각적 혼란 방지)
        if igw_exists and not has_cdn_local:
            resources["InternetGateway"] = {"Type": "AWS::EC2::InternetGateway"}
            vpc_res["BorderChildren"] = [{"Position": "S", "Resource": "InternetGateway"}]

        resources["VPC"] = vpc_res

    # ── 9. 글로벌 리소스 (VPC 외부) ─────────────────────────────────────────
    storage_names: list[str] = []
    for item in arch.get("storage", []):
        rid = item["name"]
        storage_names.append(rid)
        resources[rid] = {"Type": STORAGE_TYPE_MAP.get(item["type"], "AWS::S3::Bucket")}
        if not is_cdn_vpc:
            # CDN + VPC 아키텍처가 아닐 때만 cloud_storage에 추가
            # (CDN + VPC일 때는 S3를 cloud_entry에 배치하므로 cloud_storage에서 제외)
            cloud_storage.append(rid)

    # DNS(Route53)를 CloudFront보다 먼저 배치 (실제 통신 순서: Route53 → CloudFront)
    for item in arch.get("dns", []):
        rid = "Route53"
        resources[rid] = {"Type": "AWS::Route53::HostedZone"}
        cloud_entry.append(rid)   # 사용자 진입점 → VPC 위

    cdn_names: list[str] = []
    for item in arch.get("cdn", []):
        rid = "CloudFront"
        cdn_names.append(rid)
        resources[rid] = {"Type": "AWS::CloudFront::Distribution"}
        cloud_entry.append(rid)   # Route53 다음에 배치

    # CDN + VPC 아키텍처: S3를 cloud_entry에 포함 (CloudFront의 오리진 → VPC 위에 배치)
    if arch.get("cdn") and arch.get("vpc"):
        for item in arch.get("storage", []):
            rid = item["name"]
            if rid not in resources:  # 아직 추가 안 된 경우만
                resources[rid] = {"Type": STORAGE_TYPE_MAP.get(item["type"], "AWS::S3::Bucket")}
            cloud_entry.append(rid)

    for item in arch.get("messaging", []):
        rid = item["name"]
        resources[rid] = {"Type": MESSAGING_TYPE_MAP.get(item["type"], "AWS::SQS::Queue")}
        # SNS는 Lambda 처리 후 알림이므로 cloud_notification에 배치
        # SQS/EventBridge는 입력 큐이므로 cloud_streaming에 배치
        if item.get("type") == "SNS":
            cloud_notification.append(rid)
        else:
            cloud_streaming.append(rid)

    for item in arch.get("streaming", []):
        rid = item["name"]
        resources[rid] = {"Type": STREAMING_TYPE_MAP.get(item["type"], "AWS::Kinesis::Stream")}
        cloud_streaming.append(rid)

    for item in arch.get("auth", []):
        rid = item.get("user_pool", "CognitoUserPool")
        resources[rid] = {"Type": "AWS::Cognito::UserPool"}
        cloud_support.append(rid)  # 지원 서비스 → VPC 아래

    for item in arch.get("secrets", []):
        rid = item["name"]
        resources[rid] = {"Type": "AWS::SecretsManager::Secret"}
        cloud_support.append(rid)

    # KMS 렌더링 (security.kms: true 일 때)
    kms_exists = arch.get("security", {}).get("kms", False)
    if kms_exists:
        resources["KMS"] = {"Type": "AWS::KMS::Key"}
        cloud_support.append("KMS")

    # ── 9b. aws_cloud_children 순서 조립 ────────────────────────────────────
    # 데이터 흐름 순서:
    #   VPC + CDN: 진입점(CloudFront/Route53) → VPC → 지원서비스(ECR 등)
    #   VPC only:  VPC → 지원서비스
    #   Serverless: Streaming → Compute → Notification → DB/Storage
    aws_cloud_children: list = []

    if has_vpc:
        # VPC 아키텍처: 진입점 → VPC → SQS(큐) → Lambda(처리) → SNS(알림) → 지원 서비스
        aws_cloud_children.extend(cloud_entry)          # CloudFront, Route53 등 (VPC 위)
        aws_cloud_children.append("VPC")
        aws_cloud_children.extend(cloud_streaming)      # SQS 등 입력 큐
        aws_cloud_children.extend(cloud_compute)        # Lambda 등 처리
        aws_cloud_children.extend(cloud_notification)   # SNS 등 완료 알림
        aws_cloud_children.extend(cloud_support)        # ECR, KMS, Cognito 등
    else:
        # Serverless: 진입점 → Streaming → Compute (DB/Storage는 섹션 10에서 OutputStack으로 추가)
        aws_cloud_children.extend(cloud_entry)
        aws_cloud_children.extend(cloud_streaming)
        aws_cloud_children.extend(cloud_compute)
        aws_cloud_children.extend(cloud_notification)
        aws_cloud_children.extend(cloud_support)

    # ── 10. 레이아웃: 출력 리소스 HorizontalStack ───────────────────────────
    # CDN이 있는 VPC 아키텍처에서는 S3가 CloudFront의 오리진이므로 cloud_entry와 함께 배치
    # (S3는 OutputStack 대신 AWSCloud 상단에 CloudFront 옆에 위치)
    if has_vpc and has_cdn_local:
        output_resources = cloud_db  # S3는 제외 (이미 cloud_entry 영역에서 CloudFront와 연결됨)
    else:
        output_resources = cloud_db + cloud_storage

    if len(output_resources) > 1:
        resources["OutputStack"] = {
            "Type": "AWS::Diagram::HorizontalStack",
            "Children": output_resources,
        }
        aws_cloud_children.append("OutputStack")
    else:
        aws_cloud_children.extend(output_resources)


    # ── 11. Links (SourcePosition/TargetPosition으로 화살표 정렬) ────────────
    arrow = {"Type": "Open"}
    has_cdn = has_cdn_local
    streaming_names  = [item["name"] for item in arch.get("streaming", [])]
    original_compute = [c["name"] for c in arch.get("compute", [])]
    sqs_names        = [item["name"] for item in arch.get("messaging", []) if item.get("type") == "SQS"]
    sns_names        = [item["name"] for item in arch.get("messaging", []) if item.get("type") == "SNS"]
    ecr_names        = [item["name"] for item in arch.get("container_registry", [])]
    resources["User"] = {"Type": "AWS::Diagram::Resource", "Preset": "User"}

    if has_cdn:
        # 정적 파일: User → CloudFront
        for cdn in cdn_names:
            links.append({"Source": "User", "SourcePosition": "S",
                          "Target": cdn, "TargetPosition": "N",
                          "TargetArrowHead": arrow})
        if lb_names:
            # API 트래픽: CloudFront → ALB (CDN이 있을 때는 CloudFront를 통해 ALB로)
            links.append({"Source": cdn_names[0], "SourcePosition": "S",
                          "Target": lb_names[0], "TargetPosition": "N",
                          "TargetArrowHead": arrow})
    elif lb_names and igw_exists:
        # CDN 없이 IGW만 있는 경우: User → IGW → ALB
        links.append({"Source": "User", "SourcePosition": "S",
                      "Target": "InternetGateway", "TargetPosition": "N",
                      "TargetArrowHead": arrow})
        links.append({"Source": "InternetGateway", "SourcePosition": "S",
                      "Target": lb_names[0], "TargetPosition": "N",
                      "TargetArrowHead": arrow})
    elif lb_names:
        # IGW도 CDN도 없는 경우: User → ALB 직접
        links.append({"Source": "User", "SourcePosition": "S",
                      "Target": lb_names[0], "TargetPosition": "N",
                      "TargetArrowHead": arrow})
    elif streaming_names:
        # 스트리밍 파이프라인: User → Kinesis
        links.append({"Source": "User", "SourcePosition": "S",
                      "Target": streaming_names[0], "TargetPosition": "N",
                      "TargetArrowHead": arrow})

    # LB → Compute (원본만 연결 - Replica는 서브넷 배치로 Multi-AZ 표현)
    for lb in lb_names:
        for comp in original_compute:
            target = f"{comp}ASG" if comp in auto_scaling_map else comp
            links.append({"Source": lb, "SourcePosition": "S",
                          "Target": target, "TargetPosition": "N",
                          "TargetArrowHead": arrow})

    # Streaming → Compute (Kinesis/SQS → Lambda)
    for s_name in streaming_names + sqs_names:
        for comp in original_compute:
            comp_type = next(
                (c.get("type") for c in arch.get("compute", []) if c["name"] == comp), ""
            )
            # Lambda 워커만 연결 (ECS 웹서버는 제외)
            if comp_type == "Lambda":
                links.append({"Source": s_name, "SourcePosition": "S",
                              "Target": comp, "TargetPosition": "N",
                              "TargetArrowHead": arrow})

    # Compute → DB (원본만 연결 - Replica 제외)
    for i, comp in enumerate(original_compute):
        comp_type = next(
            (c.get("type") for c in arch.get("compute", []) if c["name"] == comp), ""
        )
        # SQS가 있는 비동기 아키텍처에서 ECS는 SQS로만 연결, DB 직접 연결 제외
        # Lambda가 DB에 저장하는 역할을 하므로 ECS→DB 화살표 불필요
        if sqs_names and comp_type in {"ECS", "EKS", "Fargate"}:
            continue
        for j, db in enumerate(db_names):
            pos = "SW" if j == 0 else "SE"
            links.append({"Source": comp, "SourcePosition": pos,
                          "Target": db, "TargetPosition": "N",
                          "TargetArrowHead": arrow})

    # Compute → Storage (CDN 없을 때만 - 원본만 연결)
    if not has_cdn:
        for comp in original_compute:
            for j, st in enumerate(storage_names):
                pos = "SE" if j == 0 else "S"
                links.append({"Source": comp, "SourcePosition": pos,
                              "Target": st, "TargetPosition": "N",
                              "TargetArrowHead": arrow})

    # API GW → Compute
    for apigw in apigw_names:
        for comp in compute_names:
            links.append({"Source": apigw, "SourcePosition": "S",
                          "Target": comp, "TargetPosition": "N",
                          "TargetArrowHead": arrow})

    # CDN → Storage
    for cdn in cdn_names:
        for st in storage_names:
            links.append({"Source": cdn, "SourcePosition": "S",
                          "Target": st, "TargetPosition": "N",
                          "TargetArrowHead": arrow})

    # ECS → SQS (비동기 작업 큐잉)
    for comp in original_compute:
        comp_type = next(
            (c.get("type") for c in arch.get("compute", []) if c["name"] == comp), ""
        )
        if comp_type in {"ECS", "EKS", "Fargate"} and sqs_names:
            links.append({"Source": comp, "SourcePosition": "S",
                          "Target": sqs_names[0], "TargetPosition": "N",
                          "TargetArrowHead": arrow})

    # ECS → ECR (컨테이너 이미지 Pull) — 원본 컴퓨팅만 (Replica 제외)
    for comp in original_compute:
        comp_type = next(
            (c.get("type") for c in arch.get("compute", []) if c["name"] == comp), ""
        )
        if comp_type in {"ECS", "EKS", "Fargate"} and ecr_names:
            links.append({"Source": comp, "SourcePosition": "E",
                          "Target": ecr_names[0], "TargetPosition": "W",
                          "TargetArrowHead": {"Type": "Open"}})

    # Lambda → SNS (작업 완료 알림)
    for comp in original_compute:
        comp_type = next(
            (c.get("type") for c in arch.get("compute", []) if c["name"] == comp), ""
        )
        if comp_type == "Lambda" and sns_names:
            links.append({"Source": comp, "SourcePosition": "S",
                          "Target": sns_names[0], "TargetPosition": "N",
                          "TargetArrowHead": arrow})

    # Lambda → KMS (암호화 키 사용)
    if kms_exists:
        for comp in original_compute:
            comp_type = next(
                (c.get("type") for c in arch.get("compute", []) if c["name"] == comp), ""
            )
            if comp_type == "Lambda":
                links.append({"Source": comp, "SourcePosition": "E",
                              "Target": "KMS", "TargetPosition": "W",
                              "TargetArrowHead": arrow})

    # ECS → NAT Gateway (프라이빗 서브넷 아웃바운드 트래픽)
    for comp in original_compute:
        comp_type = next(
            (c.get("type") for c in arch.get("compute", []) if c["name"] == comp), ""
        )
        if comp_type in {"ECS", "EKS", "Fargate", "EC2"} and nat_names:
            links.append({"Source": comp, "SourcePosition": "N",
                          "Target": nat_names[0], "TargetPosition": "S",
                          "TargetArrowHead": {"Type": "Open"}})

    # Compute → NAT Gateway 화살표는 생략
    # (NAT Gateway가 Public Subnet에 있는 것 자체가 프라이빗→인터넷 경로를 의미,
    #  화살표로 표현하면 ALB→ECS 화살표와 교차하여 가독성이 떨어짐)

    # ── 12. AWSCloud & Canvas ────────────────────────────────────────────────
    resources["AWSCloud"] = {
        "Type": "AWS::Diagram::Cloud",
        "Preset": "AWSCloudNoLogo",
        "Direction": "vertical",
        "Children": aws_cloud_children,
    }

    resources["Canvas"] = {
        "Type": "AWS::Diagram::Canvas",
        "Direction": "vertical",
        "Children": ["User", "AWSCloud"],
    }

    diagram: dict = {
        "Diagram": {
            "DefinitionFiles": [{"Type": "URL", "Url": DEFINITION_URL}],
            "Resources": resources,
        }
    }
    if links:
        diagram["Diagram"]["Links"] = links

    return diagram
