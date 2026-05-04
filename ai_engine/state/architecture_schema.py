"""AWS 아키텍처 명세 Pydantic 스키마.

LLM with_structured_output() 에 전달되어 구조화된 JSON 응답을 강제한다.
이후 yaml.dump()로 YAML 문자열로 변환된다.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Subnet(BaseModel):
    name: str
    type: Literal["public", "private"]
    cidr: str
    az: str


class VPC(BaseModel):
    cidr: str = "10.0.0.0/16"
    subnets: List[Subnet]


class NatGateway(BaseModel):
    subnet: str


class Networking(BaseModel):
    nat_gateway: Optional[List[NatGateway]] = None
    internet_gateway: Optional[bool] = None


class ComputeInstance(BaseModel):
    type: Literal["EC2", "ECS", "EKS", "Lambda", "Fargate"]
    name: str
    instance_type: Optional[str] = None
    count: Optional[int] = None
    subnets: List[str] = Field(description="List of subnet names. Use multiple subnets for EKS/ECS Multi-AZ.")


class AutoScaling(BaseModel):
    target: str
    min_capacity: int
    max_capacity: int
    metric: Literal["CPUUtilization", "RequestCount", "MemoryUtilization"] = "CPUUtilization"
    target_value: float = 70


class ContainerRegistry(BaseModel):
    type: Literal["ECR"] = "ECR"
    name: str


class ApiGateway(BaseModel):
    type: Literal["APIGateway"] = "APIGateway"
    name: str
    protocol: Literal["HTTP", "REST", "WebSocket"] = "HTTP"
    target: str


class LoadBalancer(BaseModel):
    type: Literal["ALB", "NLB"] = "ALB"
    name: str
    subnets: List[str] = Field(description="List of public subnet names. ALB requires 2+ subnets in different AZs.")


class Database(BaseModel):
    type: Literal["RDS", "DynamoDB", "ElastiCache", "OpenSearch"]
    name: str
    engine: Optional[str] = None
    instance_class: Optional[str] = None
    multi_az: bool = False
    subnets: Optional[List[str]] = Field(default=None, description="Private subnet names for the DB subnet group.")


class Cache(BaseModel):
    type: Literal["ElastiCache"] = "ElastiCache"
    engine: Literal["redis", "memcached"] = "redis"
    node_type: str = "cache.t3.micro"
    subnet: str


class Streaming(BaseModel):
    type: Literal["Kinesis", "KinesisFirehose", "MSK"]
    name: str
    shard_count: Optional[int] = None


class Messaging(BaseModel):
    type: Literal["SQS", "SNS", "EventBridge"]
    name: str
    fifo: Optional[bool] = None


class Storage(BaseModel):
    type: Literal["S3", "EFS"]
    name: str
    versioning: bool = False


class CDN(BaseModel):
    type: Literal["CloudFront"] = "CloudFront"
    origin: str


class DNS(BaseModel):
    type: Literal["Route53"] = "Route53"
    domain: str
    record_type: Literal["A", "CNAME", "ALIAS"] = "A"
    target: str


class Auth(BaseModel):
    type: Literal["Cognito"] = "Cognito"
    user_pool: str
    app_client: str


class Secrets(BaseModel):
    type: Literal["SecretsManager", "ParameterStore"]
    name: str


class Security(BaseModel):
    waf: bool = False
    shield: bool = False
    guard_duty: bool = False


class CloudWatchAlarm(BaseModel):
    metric: str
    threshold: float


class LogGroup(BaseModel):
    name: str


class Monitoring(BaseModel):
    type: Literal["CloudWatch"] = "CloudWatch"
    alarms: Optional[List[CloudWatchAlarm]] = None
    log_groups: Optional[List[LogGroup]] = None


class InboundRule(BaseModel):
    port: int
    protocol: str
    source: str


class SecurityGroup(BaseModel):
    name: str
    description: str
    inbound: List[InboundRule]


class IAMRole(BaseModel):
    name: str
    description: str
    policies: List[str]


class Architecture(BaseModel):
    name: str
    description: str
    region: str = "ap-northeast-2"
    vpc: VPC
    networking: Optional[Networking] = None
    compute: List[ComputeInstance]
    auto_scaling: Optional[List[AutoScaling]] = None
    container_registry: Optional[List[ContainerRegistry]] = None
    api_gateway: Optional[List[ApiGateway]] = None
    load_balancer: Optional[List[LoadBalancer]] = None
    database: Optional[List[Database]] = None
    cache: Optional[List[Cache]] = None
    streaming: Optional[List[Streaming]] = None
    messaging: Optional[List[Messaging]] = None
    storage: Optional[List[Storage]] = None
    cdn: Optional[List[CDN]] = None
    dns: Optional[List[DNS]] = None
    auth: Optional[List[Auth]] = None
    secrets: Optional[List[Secrets]] = None
    security: Optional[Security] = None
    monitoring: Optional[List[Monitoring]] = None
    security_groups: Optional[List[SecurityGroup]] = None
    iam: Optional[List[IAMRole]] = None


class ArchitectureSpec(BaseModel):
    """LLM with_structured_output() 루트 모델."""
    architecture: Architecture
