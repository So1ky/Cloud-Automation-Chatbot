#!/usr/bin/env bash
# 클러스터 중지: EC2 3대 stop.
#
# graceful drain은 하지 않는다 — kubeadm 클러스터의 stop→start 복원은
# 실측으로 검증됐고(프라이빗 IP 고정 + EFS 데이터 생존), 세션 종료 시
# 빠르게 끄는 것이 목적이다. EBS 65GB는 중지 중에도 과금된다(월 $5.93).
set -euo pipefail

TF_DIR="$(cd "$(dirname "$0")/../../terraform" && pwd)"

ALL_IDS=($(terraform -chdir="$TF_DIR" output -json node_instance_ids | jq -r '.[]'))

echo "▶ 인스턴스 중지: ${ALL_IDS[*]}"
aws ec2 stop-instances --instance-ids "${ALL_IDS[@]}" >/dev/null
aws ec2 wait instance-stopped --instance-ids "${ALL_IDS[@]}"
echo "완료. 3대 stopped."
