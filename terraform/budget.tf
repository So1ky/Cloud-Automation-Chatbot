# 월 비용 예산과 이메일 알림.
#
# 계정 전체 비용을 대상으로 한다(태그 필터 없음).
# 이 계정에 다른 프로젝트의 과금이 사실상 없어(2026-09-18 확인: S3 $0.00008뿐)
# 계정 전체 예산이 곧 이 프로젝트의 예산이 된다.
# 나중에 다른 프로젝트가 늘어나면 Cost Allocation Tag를 활성화하고
# cost_filter로 Project 태그를 걸어야 한다.
resource "aws_budgets_budget" "monthly" {
  name         = "${var.project_name}-monthly"
  budget_type  = "COST"
  limit_amount = var.monthly_budget_usd
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # 예측 기준 알림 — 이 추세로 가면 한 달 뒤 예산의 80%를 넘는다고 판단될 때.
  # 실제로 돈이 나가기 전에 알려주므로 "끄는 것을 잊음" 리스크에 대한 대비다.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_email]
  }

  # 실사용 기준 알림 — 이미 예산을 다 쓴 시점.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }
}
