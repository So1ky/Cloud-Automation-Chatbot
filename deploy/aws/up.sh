#!/usr/bin/env bash
# 클러스터 기동: EC2 3대 start → control-plane 새 퍼블릭 IP로 A레코드 갱신.
#
# stop/start마다 퍼블릭 IP가 바뀌므로(EIP 미사용) 기동할 때마다 이 스크립트로
# DNS를 맞춘다. 프라이빗 IP는 유지되므로 클러스터 자체는 손댈 것이 없다
# (stop→start 복원 실측 검증: 노드 Ready ~30초, 파드 전부 Running ~3분).
set -euo pipefail

TF_DIR="$(cd "$(dirname "$0")/../../terraform" && pwd)"
DOMAIN="cloudinfra.help"

# terraform state에서 식별자를 동적 조회한다 (하드코딩 방지)
INSTANCE_IDS_JSON=$(terraform -chdir="$TF_DIR" output -json node_instance_ids)
ZONE_ID=$(terraform -chdir="$TF_DIR" output -raw hosted_zone_id)
CP_ID=$(jq -r '."control-plane"' <<<"$INSTANCE_IDS_JSON")
ALL_IDS=($(jq -r '.[]' <<<"$INSTANCE_IDS_JSON"))

echo "▶ 인스턴스 시작: ${ALL_IDS[*]}"
aws ec2 start-instances --instance-ids "${ALL_IDS[@]}" >/dev/null
aws ec2 wait instance-running --instance-ids "${ALL_IDS[@]}"
echo "  3대 running 도달"

# 접속점은 지금은 control-plane 노드 IP 하나다. externalTrafficPolicy=Cluster라
# 어느 노드를 가리켜도 동작하지만, 노드 IP 직결은 단일 장애점이므로
# HA 로드맵 ③에서 헬스체크 기반 IP 선택 → 로드밸런서로 교체 예정.
CP_IP=$(aws ec2 describe-instances --instance-ids "$CP_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

aws route53 change-resource-record-sets --hosted-zone-id "$ZONE_ID" \
  --change-batch "{\"Changes\":[{\"Action\":\"UPSERT\",\"ResourceRecordSet\":{
    \"Name\":\"$DOMAIN\",\"Type\":\"A\",\"TTL\":60,
    \"ResourceRecords\":[{\"Value\":\"$CP_IP\"}]}}]}" >/dev/null
echo "▶ A레코드 갱신: $DOMAIN → $CP_IP (TTL 60)"

aws ec2 describe-instances --instance-ids "${ALL_IDS[@]}" \
  --query 'Reservations[].Instances[].[Tags[?Key==`Name`]|[0].Value, PublicIpAddress]' \
  --output table

echo "완료. 파드 복원까지 ~3분 걸린다 → https://$DOMAIN"
