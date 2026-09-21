variable "aws_region" {
  description = "리소스를 생성할 리전"
  type        = string
  default     = "ap-northeast-2" # 서울
}

variable "aws_profile" {
  description = "사용할 AWS CLI 프로파일"
  type        = string
  default     = "default"
}

variable "project_name" {
  description = "전 리소스에 붙는 태그이자 이름 접두사"
  type        = string
  default     = "cloud-automation-chatbot"
}

variable "ssh_public_key_path" {
  description = "EC2 키페어에 등록할 로컬 SSH 공개키 경로 (비상 접속용)"
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "ami_id" {
  description = <<-DESC
    노드 OS 이미지. Ubuntu 24.04 LTS (ap-northeast-2).
    최신 AMI 조회:
      aws ec2 describe-images --region ap-northeast-2 --owners 099720109477 \
        --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" \
        --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text
    최신으로 자동 추적(data source)하지 않는 이유: 새 AMI가 나올 때마다
    plan에 인스턴스 교체가 떠서 클러스터가 날아갈 위험이 있다.
  DESC
  type        = string
  default     = "ami-086a43496cb46286c" # 2026-09-04 빌드
}

variable "instance_type" {
  description = "노드 인스턴스 타입 (2 vCPU / 4 GiB)"
  type        = string
  default     = "t3.medium"
}

variable "k8s_minor_version" {
  description = "kubeadm/kubelet/kubectl 패키지 저장소의 마이너 버전. 로컬 kind 클러스터(v1.36.1)와 일치시킨다"
  type        = string
  default     = "v1.36"
}

variable "monthly_budget_usd" {
  description = "월 예산 상한(USD). 설계 예상치 $11.35에 여유를 둔 값"
  type        = string
  default     = "18"
}

variable "budget_email" {
  description = "예산 초과 알림을 받을 이메일. terraform.tfvars에 실값을 넣는다(커밋 금지)"
  type        = string
}

variable "domain_name" {
  description = "앱 도메인. Route53에서 등록했고 호스팅 존은 import로 관리한다"
  type        = string
  default     = "cloudinfra.help"
}
