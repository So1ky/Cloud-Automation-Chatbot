# Route53 — 호스팅 존과 앱 A레코드.
#
# 호스팅 존은 도메인을 Route53에서 구매할 때 자동 생성됐다(Z0827598B55NZVZONHTD).
# 여기서 apply로 새로 만들면 존이 하나 더 생기고 도메인 등록에 연결된 NS와
# 어긋나므로, 기존 존을 `terraform import`로 state에 흡수한다.
# import는 AWS에 아무것도 만들지 않고 "이 리소스는 이 코드다"라고 state에만 기록한다.
resource "aws_route53_zone" "main" {
  name = var.domain_name

  # 자동 생성 시 붙은 코멘트를 그대로 둔다. 생략하면 프로바이더 기본값
  # "Managed by Terraform"으로 바꾸려는 diff가 떠서 plan이 지저분해진다.
  comment = "HostedZone created by Route53 Registrar"
}

# 앱 진입점. worker-1(Envoy 고정 배치)의 공인 IP를 가리킨다.
# EIP를 쓰지 않으므로 IP는 stop/start마다 바뀐다 — 실값은 up.sh가 기동 시
# UPSERT로 갱신하고, Terraform은 초기값(문서용 예약 대역)만 만들고 이후 값 변화는
# ignore_changes로 무시한다. TTL 60초는 IP가 바뀌어도 1분 안에 전파되게 하기 위함.
resource "aws_route53_record" "app" {
  zone_id = aws_route53_zone.main.zone_id
  name    = var.domain_name
  type    = "A"
  ttl     = 60
  records = ["192.0.2.1"] # RFC 5737 TEST-NET-1 플레이스홀더

  lifecycle {
    ignore_changes = [records]
  }
}
