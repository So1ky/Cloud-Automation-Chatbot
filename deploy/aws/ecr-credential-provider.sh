# kubelet ECR 인증 플러그인 설치 (k8s 1.27+는 ECR 자동 인증이 제거되어 필수)
# 각 노드에서 root로 실행. 멱등 — 이미 설치돼 있으면 건너뜀.
set -e
# 1. 바이너리 설치
if [ ! -x /usr/local/bin/ecr-credential-provider ]; then
  curl -fsSL -o /usr/local/bin/ecr-credential-provider https://artifacts.k8s.io/binaries/cloud-provider-aws/v1.37.0/linux/amd64/ecr-credential-provider-linux-amd64
  chmod +x /usr/local/bin/ecr-credential-provider
fi
# 2. CredentialProviderConfig
cat > /etc/kubernetes/ecr-credential-provider-config.yaml <<'CFG'
apiVersion: kubelet.config.k8s.io/v1
kind: CredentialProviderConfig
providers:
  - name: ecr-credential-provider
    apiVersion: credentialprovider.kubelet.k8s.io/v1
    matchImages:
      - "*.dkr.ecr.*.amazonaws.com"
    defaultCacheDuration: "12h"
CFG
# 3. kubelet 플래그 추가 (멱등)
if ! grep -q image-credential-provider /var/lib/kubelet/kubeadm-flags.env; then
  sed -i 's|"$| --image-credential-provider-config=/etc/kubernetes/ecr-credential-provider-config.yaml --image-credential-provider-bin-dir=/usr/local/bin"|' /var/lib/kubelet/kubeadm-flags.env
fi
systemctl restart kubelet
sleep 3
systemctl is-active kubelet
grep -o "image-credential-provider-config=[^ ]*" /var/lib/kubelet/kubeadm-flags.env
