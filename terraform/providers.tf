# AWS 프로바이더 — Terraform이 AWS API를 호출할 때 쓰는 플러그인의 설정.
provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile

  # 이 프로바이더로 만드는 모든 리소스에 자동으로 붙는 태그.
  # 나중에 "이 프로젝트가 만든 리소스"를 찾거나 비용을 분리할 때 쓴다.
  default_tags {
    tags = {
      Project   = var.project_name
      ManagedBy = "terraform"
    }
  }
}
