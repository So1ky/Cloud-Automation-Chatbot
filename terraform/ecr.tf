# backend / frontend 이미지 저장소.
# 두 리포지토리의 설정이 완전히 같으므로 for_each로 묶는다.
resource "aws_ecr_repository" "app" {
  for_each = toset(["backend", "frontend"])

  name = "${var.project_name}/${each.key}"

  # 같은 태그로 다시 push할 수 있게 한다.
  # 실무 프로덕션은 IMMUTABLE이 표준이지만(배포 재현성), 수정 후 재배포가 잦은
  # 학습 환경에서는 태그를 매번 올리는 부담이 더 크다.
  image_tag_mutability = "MUTABLE"

  # push 시 취약점 스캔을 자동 실행한다. 기본 스캔은 무료다.
  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "${var.project_name}-${each.key}"
  }
}

# CI가 커밋마다 main-<sha> 태그로 push하므로 이미지가 무한히 쌓인다.
# 최근 10개만 남기고 만료시킨다(롤백 범위 = 최근 10회 배포). 수동 태그(0.1.x)는 영향 없음.
resource "aws_ecr_lifecycle_policy" "app" {
  for_each   = aws_ecr_repository.app
  repository = each.value.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "CI 태그(main-*) 최근 10개만 보관"
      selection = {
        tagStatus     = "tagged"
        tagPrefixList = ["main-"]
        countType     = "imageCountMoreThan"
        countNumber   = 10
      }
      action = { type = "expire" }
    }]
  })
}
