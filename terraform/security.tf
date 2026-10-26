# 쿠버네티스 노드용 보안그룹.
# 규칙은 인라인 블록이 아니라 별도 리소스로 정의한다 —
# 인라인 블록은 규칙 하나만 바꿔도 전체를 교체하므로 순간적으로 통신이 끊긴다.
resource "aws_security_group" "node" {
  name        = "${var.project_name}-node"
  description = "k8s node: intra-cluster all, external 80/443 only, no SSH (SSM)"
  vpc_id      = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-node"
  }
}

# 노드 간 통신 — 같은 보안그룹에 속한 상대에게는 모든 트래픽을 허용한다.
# API 6443, etcd 2379-2380, kubelet 10250, NodePort, CNI(VXLAN)를 일일이 열지 않아도 되고
# 포트를 빠뜨려 클러스터가 반쯤 동작하는 사고를 막는다. 외부에는 닫혀 있다.
resource "aws_vpc_security_group_ingress_rule" "node_self" {
  security_group_id            = aws_security_group.node.id
  referenced_security_group_id = aws_security_group.node.id
  ip_protocol                  = "-1" # 모든 프로토콜
  description                  = "intra-cluster: apiserver, etcd, kubelet, CNI"
}

# 외부 → HTTP. Let's Encrypt HTTP-01 챌린지와 HTTPS 리다이렉트에 필요하다.
resource "aws_vpc_security_group_ingress_rule" "http" {
  security_group_id = aws_security_group.node.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "tcp"
  from_port         = 80
  to_port           = 80
  description       = "HTTP: ACME http-01 challenge, redirect to https"
}

# 외부 → HTTPS. 실제 서비스 접속 경로다.
resource "aws_vpc_security_group_ingress_rule" "https" {
  security_group_id = aws_security_group.node.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
  description       = "HTTPS: service traffic"
}

# SSH(22)는 의도적으로 열지 않는다 — 접속은 SSM Session Manager로 한다.
# 작업 환경이 노트북이라 접속 IP가 고정되지 않고, SSM은 인바운드를 요구하지 않는다.

# 아웃바운드 전체 허용.
# ssm-agent가 SSM 엔드포인트로 나가는 연결, ECR 이미지 pull, apt 패키지 설치에 모두 필요하다.
resource "aws_vpc_security_group_egress_rule" "all" {
  security_group_id = aws_security_group.node.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
  description       = "all outbound: SSM, ECR, apt"
}
