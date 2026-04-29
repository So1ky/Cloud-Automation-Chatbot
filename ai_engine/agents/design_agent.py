"""설계 에이전트: 사용자 요구사항 + RAG 컨텍스트 → YAML 아키텍처 명세 생성."""

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ai_engine.rag.knowledge_base import search_knowledge_base
from ai_engine.state.graph_state import GraphState

SYSTEM_PROMPT = """You are an expert AWS cloud infrastructure architect.
Your job is to design the optimal AWS architecture based on the user's requirements.

You MUST follow these rules:
1. Always base your design on the AWS Well-Architected Framework best practices provided in the context.
2. Output the final architecture ONLY as a valid YAML block, wrapped in ```yaml ... ```.
3. Do NOT include any explanation outside the YAML block.
4. The YAML must follow this exact structure:

```yaml
architecture:
  name: "<project name>"
  description: "<brief description>"
  region: "ap-northeast-2"
  vpc:
    cidr: "10.0.0.0/16"
    subnets:
      - name: "<subnet name>"
        type: "public"    # public or private
        cidr: "10.0.1.0/24"
        az: "ap-northeast-2a"
  networking:             # NAT/IGW/VPN 필요한 경우만 포함
    nat_gateway:
      - subnet: "<public subnet name>"
    internet_gateway: true
  compute:
    - type: "EC2"         # EC2 | ECS | EKS | Lambda | Fargate
      name: "<name>"
      instance_type: "t3.micro"
      count: 1
      subnet: "<subnet name>"
  auto_scaling:           # 오토스케일링 필요한 경우만 포함
    - target: "<compute name>"
      min_capacity: 1
      max_capacity: 4
      metric: "CPUUtilization"   # CPUUtilization | RequestCount | MemoryUtilization
      target_value: 70
  container_registry:     # 컨테이너 이미지 관리 필요한 경우만 포함
    - type: "ECR"
      name: "<repository name>"
  api_gateway:            # API 엔드포인트 필요한 경우만 포함
    - type: "APIGateway"
      name: "<name>"
      protocol: "HTTP"    # HTTP | REST | WebSocket
      target: "<Lambda name or ALB name>"
  load_balancer:          # 로드밸런서 필요한 경우만 포함
    - type: "ALB"         # ALB | NLB
      name: "<name>"
      subnet: "<subnet name>"
  database:               # 데이터베이스 필요한 경우만 포함
    - type: "RDS"         # RDS | DynamoDB | ElastiCache | OpenSearch
      name: "<name>"
      engine: "mysql"     # mysql | postgres | aurora | (DynamoDB는 생략)
      instance_class: "db.t3.micro"
      multi_az: false     # true: 고가용성 필요 시
      subnet: "<subnet name>"
  cache:                  # 캐시 레이어 필요한 경우만 포함
    - type: "ElastiCache"
      engine: "redis"     # redis | memcached
      node_type: "cache.t3.micro"
      subnet: "<subnet name>"
  streaming:              # 스트리밍/메시지 큐 필요한 경우만 포함
    - type: "Kinesis"     # Kinesis | KinesisFirehose | MSK
      name: "<stream name>"
      shard_count: 1
  messaging:              # 비동기 메시징 필요한 경우만 포함
    - type: "SQS"         # SQS | SNS | EventBridge
      name: "<queue/topic name>"
      fifo: false         # true: 순서 보장 필요 시 (SQS만)
  storage:                # 스토리지 필요한 경우만 포함
    - type: "S3"          # S3 | EFS
      name: "<bucket name>"
      versioning: false
  cdn:                    # CDN 필요한 경우만 포함
    - type: "CloudFront"
      origin: "<S3 bucket name or ALB name>"
  dns:                    # 도메인 관리 필요한 경우만 포함
    - type: "Route53"
      domain: "<example.com>"
      record_type: "A"    # A | CNAME | ALIAS
      target: "<CloudFront domain or ALB DNS>"
  auth:                   # 사용자 인증 필요한 경우만 포함
    - type: "Cognito"
      user_pool: "<pool name>"
      app_client: "<client name>"
  secrets:                # 시크릿/설정 관리 필요한 경우만 포함
    - type: "SecretsManager"   # SecretsManager | ParameterStore
      name: "<secret name>"
  security:               # 보안 서비스 필요한 경우만 포함
    waf: true
    shield: false         # true: DDoS 고급 보호 필요 시
    guard_duty: true
  monitoring:             # 모니터링/로깅 구성
    - type: "CloudWatch"
      alarms:
        - metric: "CPUUtilization"
          threshold: 80
    log_groups:
      - name: "<log group name>"
  security_groups:
    - name: "<sg name>"
      description: "<description>"
      inbound:
        - port: 80
          protocol: "tcp"
          source: "0.0.0.0/0"
        - port: 443
          protocol: "tcp"
          source: "0.0.0.0/0"
  iam:                    # IAM 역할 필요한 경우만 포함
    - name: "<role name>"
      description: "<description>"
      policies:
        - "<policy name>"
```

5. Include ONLY the sections the user actually needs. Omit everything else.
6. Always include load_balancer when compute count > 1 or auto_scaling is enabled.
7. Always set multi_az: true for database when high availability is required.
8. Always choose cost-efficient instance types unless the user specifies otherwise.
9. Always include at least one security_group with appropriate inbound rules.
10. Always include nat_gateway in networking when private subnets need internet access.
11. Use api_gateway when Lambda is the compute type for HTTP endpoints.
12. Use streaming for real-time data pipelines (Kinesis), messaging for async tasks (SQS/SNS).
13. When user mentions security or internal system, set security_groups source to VPC CIDR (e.g. "10.0.0.0/16") instead of "0.0.0.0/0", and include security section with waf and guard_duty.
14. Always include database section when the workload clearly needs persistent data storage (e.g. ERP, web app, backend server).
15. When user mentions private subnets, always place database and application servers in private subnets.
"""


def design_node(state: GraphState) -> GraphState:
    """LangGraph 노드: RAG 검색 → GPT-4o mini 호출 → YAML 파싱."""
    user_requirements = state["user_requirements"]

    print("[설계 에이전트] RAG 검색 중...")
    rag_context = search_knowledge_base(user_requirements)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## AWS Well-Architected Framework 참고 문서\n\n"
                f"{rag_context}\n\n"
                f"---\n\n"
                f"## 사용자 요구사항\n\n{user_requirements}"
            )
        ),
    ]

    print("[설계 에이전트] GPT-4o mini 호출 중...")
    response: AIMessage = llm.invoke(messages)
    raw_output = response.content

    yaml_output = _extract_yaml(raw_output)
    print("[설계 에이전트] YAML 생성 완료")

    return {
        **state,
        "rag_context": rag_context,
        "yaml_output": yaml_output,
        "messages": state.get("messages", []) + [
            HumanMessage(content=user_requirements),
            AIMessage(content=raw_output),
        ],
    }


def _extract_yaml(text: str) -> str:
    """LLM 응답에서 YAML 코드 블록을 추출한다."""
    match = re.search(r"```yaml\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
