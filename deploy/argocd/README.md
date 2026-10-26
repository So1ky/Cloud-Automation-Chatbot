# Argo CD 부트스트랩 (1회, control-plane에서)

설계: `docs/superpowers/specs/2026-10-26-cicd-design.md`(로컬). 버전 **v3.5.6**.
Argo CD 자체는 Argo가 관리하지 않는다(설치·이 디렉터리의 파일은 수동 apply).

## 1. 설치

```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/v3.5.6/manifests/install.yaml
kubectl -n argocd rollout status deploy/argocd-server --timeout=300s
```

메모리가 빠듯하면(워커 4GB):
`kubectl -n argocd scale deploy argocd-dex-server argocd-notifications-controller argocd-applicationset-controller --replicas=0`
(내장 admin 로그인이라 dex 불필요)

## 2. 공개 (Gateway 뒤 평문)

```bash
kubectl apply -f deploy/argocd/cmd-params.yaml
kubectl -n argocd rollout restart deploy/argocd-server
kubectl apply -f deploy/argocd/httproute.yaml
kubectl -n cac get certificate   # argocd-cloudinfra-help-tls READY=True 까지 ~1분
```

브라우저: https://argocd.cloudinfra.help

## 3. admin 비밀번호

```bash
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d; echo
```

UI 로그인 → User Info → Update Password 후:
`kubectl -n argocd delete secret argocd-initial-admin-secret`

## 4. Application 등록 — 먼저 수동 모드로 diff 확인

```bash
sed '/^  syncPolicy:/,$d' deploy/argocd/application.yaml | kubectl apply -f -
```

UI에서 `cac` 앱 → App Diff로 git↔클러스터 차이를 확인. 예상 밖의 차이(특히 PV/PVC/StorageClass)는
git 쪽을 고쳐 다시 push한다. 납득되면 Sync → Synced/Healthy 확인 후 자동 전환:

```bash
kubectl apply -f deploy/argocd/application.yaml
```

## 5. (선택) GitHub 웹훅 — 폴링 3분 대신 즉시 동기화

```bash
SECRET=$(openssl rand -hex 20); echo "$SECRET"
kubectl -n argocd patch secret argocd-secret -p "{\"stringData\":{\"webhook.github.secret\":\"$SECRET\"}}"
```

GitHub 저장소 Settings → Webhooks → Add: Payload URL `https://argocd.cloudinfra.help/api/webhook`,
Content type `application/json`, Secret 위 값, 이벤트 `Just the push event`.

## 운영 메모

- 클러스터를 꺼두면 Argo도 꺼진다. cac-up 후 쌓인 main 커밋을 자동으로 따라잡는다.
- 롤백: 봇 커밋(`chore(deploy): 이미지 태그 ...`)을 `git revert` → main push → Argo가 이전 태그로 복귀.
- `secrets.yaml`은 Argo 범위 밖. 값 변경 시 `kubectl apply -f deploy/k8s/secrets.yaml` + rollout restart.
