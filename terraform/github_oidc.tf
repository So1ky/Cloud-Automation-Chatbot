# GitHub Actions → AWS 자격증명 (OIDC).
#
# 액세스 키를 GitHub Secret에 넣는 대신, Actions가 매 실행마다 GitHub에서 받은
# 단기 토큰(JWT)을 AWS STS에 제시해 역할을 빌려 쓴다. 저장되는 비밀이 없다.
# AWS는 GitHub의 발급자 URL을 신뢰하도록 "OIDC 공급자"를 한 번 등록해야 한다.
resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]

  # GitHub 인증서 체인의 지문. AWS는 2023년부터 이 발급자를 자체 신뢰 저장소로
  # 검증하므로 값이 바뀌어도 동작하지만, 필드가 필수라 공식 공개 값을 적는다.
  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1",
    "1c58a3a8518e8759bf075b76b750d4f2df264fcd",
  ]
}

# Actions가 빌려 쓸 역할. 신뢰 정책의 sub 조건이 핵심 —
# 이 저장소의 main 브랜치에서 실행된 워크플로만 통과한다.
# (PR·포크·다른 브랜치는 sub 값이 달라 거부된다)
resource "aws_iam_role" "github_actions" {
  name = "${var.project_name}-github-actions"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRoleWithWebIdentity"
      Principal = {
        Federated = aws_iam_openid_connect_provider.github.arn
      }
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          "token.actions.githubusercontent.com:sub" = "repo:${var.github_repo}:ref:refs/heads/main"
        }
      }
    }]
  })

  tags = {
    Name = "${var.project_name}-github-actions"
  }
}

# ECR push에 필요한 최소 권한. GetAuthorizationToken은 리소스 지정이 불가능해 "*".
# 나머지는 이 프로젝트의 두 리포지토리에만 허용한다.
resource "aws_iam_role_policy" "github_actions_ecr_push" {
  name = "ecr-push"
  role = aws_iam_role.github_actions.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
        ]
        Resource = [for r in aws_ecr_repository.app : r.arn]
      },
    ]
  })
}
