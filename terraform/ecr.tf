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
