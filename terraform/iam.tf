# EC2가 빌려 쓸(assume) IAM 역할.
#
# assume_role_policy는 "이 역할에 무엇을 허용하는가"가 아니라
# "누가 이 역할을 빌려 쓸 수 있는가"를 정의한다(신뢰 정책).
# 여기서는 EC2 서비스만 이 역할을 맡을 수 있다.
resource "aws_iam_role" "node" {
  name = "${var.project_name}-node"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = {
    Name = "${var.project_name}-node"
  }
}

# ECR에서 이미지를 pull할 권한. push는 로컬에서 하므로 읽기 전용으로 충분하다.
resource "aws_iam_role_policy_attachment" "ecr_read" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# Session Manager 접속 권한. SSH 22번을 열지 않는 대가로 필요하다.
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# EBS CSI 드라이버 권한 — postgres PVC용 EBS 볼륨을 클러스터가 직접
# 생성·attach·삭제할 수 있어야 한다. CSI 드라이버 파드는 노드의 인스턴스
# 프로파일 자격증명을 물려받으므로 노드 역할에 붙인다.
resource "aws_iam_role_policy_attachment" "ebs_csi" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
}

# Instance Profile — IAM 역할을 EC2에 끼우기 위한 어댑터.
# EC2는 역할을 직접 붙일 수 없어 이 껍데기를 거쳐야 한다.
resource "aws_iam_instance_profile" "node" {
  name = "${var.project_name}-node"
  role = aws_iam_role.node.name
}
