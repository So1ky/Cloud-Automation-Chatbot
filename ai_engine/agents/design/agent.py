"""설계 에이전트: 사용자 요구사항 + RAG 컨텍스트 → 구조화된 아키텍처 명세 생성."""

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import APIConnectionError, APITimeoutError, RateLimitError

from ai_engine.config import get_llm
from ai_engine.rag.knowledge_base import search_knowledge_base
from ai_engine.state.architecture_schema import ArchitectureSpec
from ai_engine.state.graph_state import GraphState

SYSTEM_PROMPT = """You are an expert AWS cloud infrastructure architect.
Your job is to design the optimal AWS architecture based on the user's requirements.

You MUST follow these rules:

General rules:
1. Always base your design on the AWS Well-Architected Framework best practices provided in the context.
2. Fill in ALL required fields. For optional sections, include them ONLY when the user's requirements actually need them.
3. The region is always "ap-northeast-2" unless the user specifies otherwise.

VPC usage rules:
4. ONLY create a VPC when the architecture includes EC2, ECS, EKS, or Fargate resources that must run inside a private network.
5. Do NOT create a VPC for the following cases — leave vpc as null, no subnets, no internet_gateway, no nat_gateway:
   - Static website hosting: S3 + CloudFront + Route 53 only
   - Serverless event pipeline: Lambda + Kinesis/SQS/SNS + S3/DynamoDB only
   - Any architecture that consists entirely of managed services: S3, CloudFront, Route 53, Lambda, DynamoDB, API Gateway, Cognito, SNS, SQS, Kinesis
   Lambda does NOT need a VPC to access Kinesis, S3, DynamoDB, or SQS — these are all public AWS endpoints.
   Adding Lambda to a VPC without a NAT Gateway or VPC Endpoint will BREAK connectivity to these services.

Subnet assignment rules:
6. load_balancer.subnets must contain 2+ PUBLIC subnet names in different AZs (ALB requires multi-AZ).
7. compute.subnets for EKS/ECS must contain 2+ PRIVATE subnet names in different AZs for high availability.
   NEVER place ECS/EKS/Fargate in a public subnet. Only ALB and NAT Gateway belong in public subnets.
8. compute.subnets for EC2/Lambda can be a single subnet.
9. database.subnets must always list PRIVATE subnet names only.
10. Always include nat_gateway in networking when private subnets need internet access.
    nat_gateway MUST be placed in a PUBLIC subnet (e.g. subnet: "PublicSubnet1").
    NEVER place nat_gateway in a private subnet. NAT Gateway needs to reach the Internet Gateway.

Auto Scaling rules:
11. auto_scaling.target must EXACTLY match the name field of one of the compute resources defined in the compute list.
    Example: if compute name is "AppServer", then auto_scaling target must be "AppServer", not a port number.
12. Always include load_balancer when auto_scaling is enabled or compute count > 1.

Security Group rules:
13. When external access is needed (e.g. public-facing web service), create SEPARATE security groups:
    - One for the ALB (source: "0.0.0.0/0", ports 80/443) for external traffic
    - One for internal compute/DB resources (source: VPC CIDR e.g. "10.0.0.0/16") for internal traffic
14. When the user mentions security, internal system, or private-only access, set ALL security group sources to VPC CIDR and include waf=true and guard_duty=true in security section.

Service placement rules:
15. Always include database section when the workload clearly needs persistent data storage (e.g. ERP, web app, backend server).
16. DynamoDB must ALWAYS be placed in the database section, NEVER in storage. storage only allows S3 or EFS.
17. Use api_gateway ONLY when Lambda handles HTTP/REST requests from external clients.
    Do NOT add api_gateway when compute includes ECS, EKS, EC2, or Fargate with a load_balancer.
    For ECS/EKS + ALB architectures, traffic flows CloudFront → ALB directly. No API Gateway needed.
18. Use streaming for real-time data pipelines (Kinesis), messaging for async tasks (SQS/SNS).
19. Always set multi_az: true for database when high availability is required.
20. Always choose cost-efficient instance types unless the user specifies otherwise.

Frontend/Static file optimization rules:
21. When the user mentions React, Vue, Angular, Next.js, or any SPA/frontend framework:
    - You MUST include BOTH storage (S3) AND cdn (CloudFront) in the output. This is mandatory.
    - ALWAYS serve frontend static files via S3 (storage) + CloudFront (cdn), NOT via ECS/EC2/compute.
    - ECS/EC2 compute resources should handle ONLY the backend API server.
    - Example: "React frontend + Node.js backend on ECS"
      → storage: [{name: "FrontendBucket", type: "S3"}]
      → cdn: [{name: "CloudFront", origin: "FrontendBucket"}]
      → compute: [{name: "NodejsBackend", type: "ECS", subnets: ["PrivateSubnet1", "PrivateSubnet2"]}]
    - VPC is still required for the ECS backend in this case.
    - NEVER omit S3 when React/Vue/Angular is mentioned. S3 is the mandatory origin for CloudFront.
22. Only use compute for server-side rendered apps (e.g. Next.js SSR) if rendering must happen server-side.


Async processing rules:
23. When the user mentions tasks that take a long time (e.g. AI processing, video encoding, batch jobs, background tasks):
    - ALWAYS use SQS (messaging) as a queue between the web server and the worker.
    - ALWAYS add a Lambda function (compute, type: Lambda, subnets: null) as the async worker.
    - NEVER generate SQS without a corresponding Lambda worker.
    - Flow: ECS (web, private subnet) → SQS → Lambda (no VPC) → storage/database
24. When the user mentions "notification", "push alert", "알림", or "notify when done":
    - ALWAYS add SNS (messaging) to send the completion notification.
    - Flow: Lambda/Worker → SNS → User

Security and encryption rules:
25. When the user mentions "encrypt", "암호화", or "secure storage of sensitive data":
    - ALWAYS add KMS to the security section (kms: true).
    - KMS is used to encrypt data stored in DynamoDB, S3, or RDS.
26. When the user mentions "login", "회원", "auth", or "user account":
    - ALWAYS add Cognito to the auth section.

Global service rules:
27. When the user mentions "global", "worldwide", "전 세계", or multiple regions/countries:
    - ALWAYS include cdn (CloudFront) and dns (Route 53) for global content delivery.
    - Set DynamoDB multi_az: true or mention Global Table in the architecture.

Data flow design rules (for clean diagram readability):
28. Always design with a clear top-to-bottom data flow:
    - Entry point at top: User → DNS/CDN/ALB
    - Processing in middle: ECS/Lambda
    - Storage at bottom: RDS/DynamoDB/S3
29. Include ONLY the components that are in the actual data path. Do not add services
    unless the user explicitly needs them (e.g. do NOT add Cognito unless auth is required,
    do NOT add ECR unless container registry is mentioned).
30. Keep architectures minimal and focused. Fewer components = cleaner diagram.
    Only add monitoring (CloudWatch), security (WAF, GuardDuty), or messaging (SQS)
    when the user explicitly requests those features.

The YAML structure follows these types:
- compute.type: EC2 | ECS | EKS | Lambda | Fargate
- database.type: RDS | DynamoDB | ElastiCache | OpenSearch
- storage.type: S3 | EFS  (DynamoDB is NOT a storage type)
- messaging.type: SQS | SNS | EventBridge
- streaming.type: Kinesis | KinesisFirehose | MSK
"""


def design_node(state: GraphState) -> dict:
    """LangGraph 노드: RAG 검색 → GPT-4o mini (structured output) → YAML 변환.

    검증 에이전트가 아키텍처 문제로 되돌려 보낸 경우(feedback 존재)에는
    이전 YAML과 검증 피드백을 함께 전달해 설계를 수정하도록 한다.
    """
    user_requirements = state["user_requirements"]
    feedback = state.get("feedback", "")
    previous_yaml = state.get("yaml_output", "")
    is_retry = bool(feedback) and state.get("fix_target") == "design"

    # 재시도가 아닐 때만 RAG 검색 (재시도 시 기존 컨텍스트 재사용)
    rag_context = state.get("rag_context", "")
    if not is_retry or not rag_context:
        print("[설계 에이전트] RAG 검색 중...")
        try:
            rag_context = search_knowledge_base(user_requirements)
        except Exception as e:
            print(f"[설계 에이전트] RAG 검색 실패: {e}")
            rag_context = "Well-Architected Framework 문서를 검색하지 못했습니다."

    llm = get_llm("design")
    structured_llm = llm.with_structured_output(ArchitectureSpec)

    human_content = (
        f"## AWS Well-Architected Framework 참고 문서\n\n"
        f"{rag_context}\n\n"
        f"---\n\n"
        f"## 사용자 요구사항\n\n{user_requirements}"
    )
    if is_retry:
        print("[설계 에이전트] 검증 피드백 반영하여 재설계 중...")
        human_content += (
            f"\n\n---\n\n"
            f"## 이전 설계 (검증 실패)\n\n{previous_yaml}\n\n"
            f"## 검증 에이전트 피드백\n\n{feedback}\n\n"
            f"## 재설계 지시 (반드시 준수)\n"
            f"1. 피드백에서 지적된 문제를 해결하는 데 필요한 최소한의 변경만 하세요.\n"
            f"2. 문제가 없는 부분은 이전 설계를 그대로 유지하세요.\n"
            f"3. 피드백이 명시적으로 요구하지 않는 한 새로운 AWS 서비스, 도메인, 인증서를 "
            f"추가하지 마세요. 복잡도를 늘리는 방향이 아니라 단순화하는 방향으로 해결하세요."
        )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    print("[설계 에이전트] GPT-4o mini 호출 중...")
    try:
        spec: ArchitectureSpec = structured_llm.invoke(messages)
    except RateLimitError as e:
        raise RuntimeError(f"OpenAI API 요청 한도 초과: {e}") from e
    except APITimeoutError as e:
        raise RuntimeError(f"OpenAI API 타임아웃: {e}") from e
    except APIConnectionError as e:
        raise RuntimeError(f"OpenAI API 연결 실패: {e}") from e
    except Exception as e:
        raise RuntimeError(f"LLM 호출 중 오류 발생: {e}") from e

    spec_dict = spec.model_dump(exclude_none=True)
    yaml_output = yaml.dump(spec_dict, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print("[설계 에이전트] YAML 생성 완료")

    # LangGraph 노드는 변경된 키만 반환한다.
    # messages는 add_messages reducer가 자동으로 기존 히스토리에 append한다.
    return {
        "rag_context": rag_context,
        "yaml_output": yaml_output,
        "messages": [
            HumanMessage(content=user_requirements),
            AIMessage(content=yaml_output),
        ],
    }
