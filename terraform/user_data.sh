#!/bin/bash
# EC2 최초 부팅 시 1회 실행된다. 로그: /var/log/cloud-init-output.log
# 이 스크립트는 클러스터를 만들지 않는다 — kubeadm이 요구하는 "재료"만 깔아둔다.
set -euxo pipefail

# ── 1. swap 비활성화 ─────────────────────────────────────────
# kubeadm은 swap이 켜져 있으면 기동을 거부한다.
# kubelet의 메모리 관리(파드 축출 판단)가 swap을 전제하지 않기 때문이다.
swapoff -a
sed -i '/ swap / s/^/#/' /etc/fstab

# ── 2. 커널 모듈 ─────────────────────────────────────────────
# overlay      : 컨테이너 이미지 레이어를 쌓는 overlayfs
# br_netfilter : 브리지를 지나는 트래픽을 iptables가 볼 수 있게 한다 (파드 간 통신 제어에 필요)
cat > /etc/modules-load.d/k8s.conf <<'MODULES'
overlay
br_netfilter
MODULES
modprobe overlay
modprobe br_netfilter

# ── 3. sysctl ────────────────────────────────────────────────
# ip_forward가 꺼져 있으면 노드가 파드 트래픽을 전달하지 못한다.
cat > /etc/sysctl.d/k8s.conf <<'SYSCTL'
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
SYSCTL
sysctl --system

# ── 4. containerd (컨테이너 런타임) ──────────────────────────
export DEBIAN_FRONTEND=noninteractive
apt-get update
# nfs-common: EFS(rag-assets)를 기본 nfs 타입 PV로 마운트할 때 노드에 필요하다.
apt-get install -y containerd apt-transport-https ca-certificates curl gpg unzip nfs-common

mkdir -p /etc/containerd
containerd config default > /etc/containerd/config.toml
# cgroup 드라이버를 systemd로 맞춘다.
# kubelet의 기본값이 systemd이고, 둘이 어긋나면 파드가 원인 불명으로 재시작한다.
# kubeadm 설치에서 가장 흔한 실패 원인이다.
sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
systemctl restart containerd
systemctl enable containerd

# ── 5. kubeadm / kubelet / kubectl ───────────────────────────
mkdir -p /etc/apt/keyrings
curl -fsSL "https://pkgs.k8s.io/core:/stable:/${k8s_minor}/deb/Release.key" \
  | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/${k8s_minor}/deb/ /" \
  > /etc/apt/sources.list.d/kubernetes.list

apt-get update
apt-get install -y kubelet kubeadm kubectl
# apt 자동 업그레이드로 노드 간 버전이 어긋나는 것을 막는다.
apt-mark hold kubelet kubeadm kubectl

# ── 6. AWS CLI (ECR 로그인 확인용) ──────────────────────────
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install
rm -rf /tmp/aws /tmp/awscliv2.zip

# ── 완료 표시 ────────────────────────────────────────────────
# 검증할 때 이 파일의 존재로 성공을 판단한다.
echo "$(date -Is) bootstrap complete: k8s ${k8s_minor}" > /var/log/k8s-bootstrap-done
