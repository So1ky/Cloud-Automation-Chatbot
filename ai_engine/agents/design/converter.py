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
    "DynamoDB":    "AWS::DynamoDB::Table",
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
    aws_cloud_children: list = []

    # 서브넷별 자식 리소스 추적 (name → [resource_id, ...])
    subnet_children: dict = {
        s["name"]: [] for s in arch.get("vpc", {}).get("subnets", [])
    }
    subnet_type_map: dict = {
        s["name"]: s["type"] for s in arch.get("vpc", {}).get("subnets", [])
    }

    # ── 1. Compute ──────────────────────────────────────────────────────────
    compute_names: list[str] = []
    for item in arch.get("compute", []):
        rid = item["name"]
        compute_names.append(rid)
        resources[rid] = {"Type": COMPUTE_TYPE_MAP.get(item["type"], "AWS::EC2::Instance")}
        for sn in item.get("subnets", []):
            if sn in subnet_children:
                subnet_children[sn].append(rid)

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
        for sn in item.get("subnets", []):
            if sn in subnet_children:
                subnet_children[sn].append(rid)

    # ── 3. Database ─────────────────────────────────────────────────────────
    db_names: list[str] = []
    for item in arch.get("database", []):
        rid = item["name"]
        db_names.append(rid)
        resources[rid] = {"Type": DATABASE_TYPE_MAP.get(item["type"], "AWS::RDS::DBInstance")}
        for sn in item.get("subnets", []):
            if sn in subnet_children:
                subnet_children[sn].append(rid)

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
        aws_cloud_children.append(rid)

    # ── 6. Container Registry ───────────────────────────────────────────────
    for item in arch.get("container_registry", []):
        rid = item["name"]
        resources[rid] = {"Type": "AWS::ECR::Repository"}
        aws_cloud_children.append(rid)

    # ── 7. NAT Gateway & Internet Gateway ──────────────────────────────────
    networking = arch.get("networking") or {}
    igw_exists = networking.get("internet_gateway", False)
    nat_names: list[str] = []

    for i, nat in enumerate(networking.get("nat_gateway") or []):
        rid = f"NATGateway{i + 1}" if i > 0 else "NATGateway"
        nat_names.append(rid)
        resources[rid] = {"Type": "AWS::EC2::NatGateway"}
        sn = nat.get("subnet", "")
        if sn in subnet_children:
            subnet_children[sn].append(rid)

    # ── 8. 서브넷 → VPC 구성 ────────────────────────────────────────────────
    public_subnets  = [n for n, t in subnet_type_map.items() if t == "public"]
    private_subnets = [n for n, t in subnet_type_map.items() if t == "private"]

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

    vpc_res: dict = {
        "Type": "AWS::EC2::VPC",
        "Direction": "vertical",
        "Children": vpc_children,
    }
    if igw_exists:
        resources["InternetGateway"] = {"Type": "AWS::EC2::InternetGateway"}
        vpc_res["BorderChildren"] = [{"Position": "S", "Resource": "InternetGateway"}]

    resources["VPC"] = vpc_res
    aws_cloud_children.insert(0, "VPC")

    # ── 9. 글로벌 리소스 (VPC 외부) ─────────────────────────────────────────
    storage_names: list[str] = []
    for item in arch.get("storage", []):
        rid = item["name"]
        storage_names.append(rid)
        resources[rid] = {"Type": STORAGE_TYPE_MAP.get(item["type"], "AWS::S3::Bucket")}
        aws_cloud_children.append(rid)

    cdn_names: list[str] = []
    for item in arch.get("cdn", []):
        rid = f"CloudFront"
        cdn_names.append(rid)
        resources[rid] = {"Type": "AWS::CloudFront::Distribution"}
        aws_cloud_children.append(rid)

    for item in arch.get("dns", []):
        rid = "Route53"
        resources[rid] = {"Type": "AWS::Route53::HostedZone"}
        aws_cloud_children.append(rid)

    for item in arch.get("messaging", []):
        rid = item["name"]
        resources[rid] = {"Type": MESSAGING_TYPE_MAP.get(item["type"], "AWS::SQS::Queue")}
        aws_cloud_children.append(rid)

    for item in arch.get("streaming", []):
        rid = item["name"]
        resources[rid] = {"Type": STREAMING_TYPE_MAP.get(item["type"], "AWS::Kinesis::Stream")}
        aws_cloud_children.append(rid)

    for item in arch.get("auth", []):
        rid = item.get("user_pool", "CognitoUserPool")
        resources[rid] = {"Type": "AWS::Cognito::UserPool"}
        aws_cloud_children.append(rid)

    for item in arch.get("secrets", []):
        rid = item["name"]
        resources[rid] = {"Type": "AWS::SecretsManager::Secret"}
        aws_cloud_children.append(rid)

    # ── 10. Links ────────────────────────────────────────────────────────────
    # IGW → LB
    if igw_exists and lb_names:
        links.append({"Source": "InternetGateway", "Target": lb_names[0],
                      "TargetArrowHead": {"Type": "Open"}})

    # LB → Compute
    for lb in lb_names:
        for comp in compute_names:
            links.append({"Source": lb, "Target": comp,
                          "TargetArrowHead": {"Type": "Open"}})

    # Compute → DB
    for comp in compute_names:
        for db in db_names:
            links.append({"Source": comp, "Target": db,
                          "TargetArrowHead": {"Type": "Open"}})

    # Compute → Storage
    for comp in compute_names:
        for st in storage_names:
            links.append({"Source": comp, "Target": st,
                          "TargetArrowHead": {"Type": "Open"}})

    # API GW → Compute (Lambda)
    for apigw in apigw_names:
        for comp in compute_names:
            links.append({"Source": apigw, "Target": comp,
                          "TargetArrowHead": {"Type": "Open"}})

    # CDN → Storage
    for cdn in cdn_names:
        for st in storage_names:
            links.append({"Source": cdn, "Target": st,
                          "TargetArrowHead": {"Type": "Open"}})

    # Streaming → Compute
    for item in arch.get("streaming", []):
        for comp in compute_names:
            links.append({"Source": item["name"], "Target": comp,
                          "TargetArrowHead": {"Type": "Open"}})

    # ── 11. AWSCloud & Canvas ────────────────────────────────────────────────
    resources["AWSCloud"] = {
        "Type": "AWS::Diagram::Cloud",
        "Preset": "AWSCloudNoLogo",
        "Direction": "vertical",
        "Children": aws_cloud_children,
    }

    resources["Canvas"] = {
        "Type": "AWS::Diagram::Canvas",
        "Direction": "vertical",
        "Children": ["AWSCloud"],
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
