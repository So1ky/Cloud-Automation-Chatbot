# Terraform 본체와 프로바이더의 버전 범위를 고정한다.
# 버전이 달라지면 같은 코드가 다른 동작을 할 수 있으므로 명시한다.
terraform {
  required_version = ">= 1.12"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.65" # 6.65 이상 7.0 미만 (호환되는 마이너 업데이트는 허용)
    }
  }
}
