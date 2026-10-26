# EFS — rag-assets(ChromaDB 인덱스 등 148MB) 저장용 공유 파일시스템.
#
# 왜 EFS인가: local-path는 노드 루트 EBS 안의 디렉토리라 노드가 죽으면
# 데이터도 같이 접근 불가가 된다. EFS는 노드·클러스터와 수명이 분리돼
# 클러스터를 부수고 다시 만들어도 데이터가 남는다. RWX(여러 노드 동시 마운트)도 된다.
# postgres는 여기 두지 않는다 — NFS 위의 DB는 파일 잠금·fsync 문제로 손상 위험이
# 있어 EBS CSI 볼륨을 쓴다(iam.tf의 정책 attachment 참고).
resource "aws_efs_file_system" "rag_assets" {
  # One Zone: 단일 AZ 저장. 표준(리전) 대비 저장 단가가 절반 이하다.
  # 어차피 노드 전부가 ap-northeast-2a 한 곳에 있어 다중 AZ 내구성이 무의미하다.
  availability_zone_name = "${var.aws_region}a"

  encrypted = true

  tags = {
    Name = "${var.project_name}-rag-assets"
  }
}

# 마운트 타겟 — EFS를 서브넷에 노출하는 ENI(네트워크 인터페이스).
# 노드는 이 ENI의 IP로 NFS(2049) 연결을 맺는다. 노드와 같은 SG를 붙이면
# security.tf의 self-referencing 규칙이 "같은 SG끼리 모든 트래픽 허용"이므로
# 2049 포트를 따로 열 필요가 없다.
resource "aws_efs_mount_target" "rag_assets" {
  file_system_id  = aws_efs_file_system.rag_assets.id
  subnet_id       = aws_subnet.public.id
  security_groups = [aws_security_group.node.id]
}
