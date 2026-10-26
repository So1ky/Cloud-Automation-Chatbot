# 배포 가이드 (Kubernetes)

Cloud-Automation-Chatbot을 쿠버네티스 클러스터에 배포하는 절차입니다.

## 구성

| 구성요소 | 이미지 | 역할 |
|---|---|---|
| backend | ECR `cloud-automation-chatbot/backend` | FastAPI + ai_engine (LangGraph 3에이전트) |
| frontend | ECR `cloud-automation-chatbot/frontend` | Next.js 16 (standalone 번들) |
| postgres | `postgres:17-alpine` | 애플리케이션 DB (StatefulSet) |
| gateway | Envoy Gateway | `/` → frontend, `/api` → backend 동일 오리진 라우팅 |

이미지는 **linux/amd64** 단일 플랫폼이며, 태그는 `deploy/k8s/kustomization.yaml`의 `images`가 관리합니다
(CI가 빌드 후 자동 갱신 — 아래 "CI/CD" 참고).

## 사전 요구사항

- 쿠버네티스 클러스터 (검증 환경: kind v1.36.1, 3노드) + `kubectl`
- `helm` (Envoy Gateway 설치용)
- StorageClass — 매니페스트는 AWS EC2 클러스터 기준(`ebs-gp3` + EFS static PV)입니다. kind 등
  다른 환경에서는 `kustomization.yaml`의 `storageclass-ebs.yaml`·`efs-rag-assets.yaml`을 주석 처리하고
  `kind/nfs-fallback.yaml`을 먼저 apply한 뒤 `postgres.yaml`의 `storageClassName`을 맞춰 수정
- `OPENAI_API_KEY` (필수), `INFRACOST_CLI_AUTHENTICATION_TOKEN` (비용 분석용)

## 배포 순서

### 1. Envoy Gateway 설치

```bash
helm install eg oci://docker.io/envoyproxy/gateway-helm --version v1.9.1 \
  -n envoy-gateway-system --create-namespace
kubectl wait --for=condition=Available -n envoy-gateway-system deploy/envoy-gateway --timeout=300s
```

### 2. 네임스페이스 · 시크릿 · 설정

```bash
kubectl apply -f deploy/k8s/namespace.yaml

cp deploy/k8s/secrets.example.yaml deploy/k8s/secrets.yaml
# secrets.yaml의 값을 채운다 (OPENAI_API_KEY, SECRET_KEY, POSTGRES_PASSWORD, DATABASE_URL, INFRACOST 토큰)
#   SECRET_KEY 생성: python -c "import secrets; print(secrets.token_hex(32))"
#   DATABASE_URL의 비밀번호는 POSTGRES_PASSWORD와 반드시 동일하게
kubectl apply -f deploy/k8s/secrets.yaml

kubectl apply -f deploy/k8s/configmap.yaml
```

> `secrets.yaml`은 실제 키가 들어가므로 `.gitignore`에 등록되어 있습니다. 커밋 금지.

### 3. 볼륨 · DB · 애플리케이션

```bash
kubectl apply -k deploy/k8s   # kustomization.yaml에 나열된 전부 (StorageClass·PV/PVC·DB·앱·게이트웨이)
```

운영 클러스터에서는 이 단계를 Argo CD가 대신합니다(아래 "CI/CD").

DB 마이그레이션은 backend 기동 시 자동 적용되므로 별도 작업이 필요 없습니다.

### 4. RAG 자산 주입 (필수)

RAG 지식 베이스(ChromaDB 인덱스 + AWS Well-Architected PDF, 약 180MB)는 **이미지에 포함되지
않고 PVC로 공급**됩니다. 주입하지 않으면 챗봇은 기동하지만 RAG 검색이 빈 결과를 반환합니다.

저장소 루트에서 실행합니다. backend 파드가 `Running`이면 (`Ready` 이전이어도) 복사할 수 있습니다.

```bash
POD=$(kubectl get pod -n cac -l app=backend -o jsonpath='{.items[0].metadata.name}')

kubectl cp ai_engine/rag/chroma_db "cac/$POD:/app/ai_engine/rag/"
kubectl cp ai_engine/docs          "cac/$POD:/app/ai_engine/"

# 확인 — chroma.sqlite3와 PDF 7개가 보여야 함
kubectl exec -n cac "$POD" -- ls /app/ai_engine/rag/chroma_db /app/ai_engine/docs

# 인덱스를 새로 읽도록 재기동
kubectl rollout restart -n cac deploy/backend
```

두 경로는 `rag-assets` PVC의 `subPath` 마운트라, 파드로 복사하면 볼륨에 영속 저장됩니다.

### 5. 기동 확인 및 접속

```bash
kubectl wait --for=condition=Ready -n cac pod -l app=backend --timeout=600s
kubectl get pods -n cac
```

backend warmup(Chroma + BM25 + CrossEncoder 로딩)에 1~2분 걸리므로 `startupProbe`가 최대 5분
대기합니다. 접속 주소는 게이트웨이 서비스에서 확인합니다.

```bash
kubectl get svc -n envoy-gateway-system -l gateway.envoyproxy.io/owning-gateway-name=cac-gateway
```

`EXTERNAL-IP`가 할당되지 않는 환경(LoadBalancer 미지원)에서는 포트포워딩으로 접속합니다.

```bash
kubectl port-forward -n envoy-gateway-system \
  svc/$(kubectl get svc -n envoy-gateway-system -l gateway.envoyproxy.io/owning-gateway-name=cac-gateway -o jsonpath='{.items[0].metadata.name}') 8080:80
# → http://localhost:8080
```

접속 주소가 `http://localhost`가 아니면 `configmap.yaml`의 `FRONTEND_URL`을 실제 주소로
맞춰주세요 (인증 쿠키/리다이렉트 기준).

## 이미지 직접 빌드 (선택)

```bash
docker buildx build --platform linux/amd64 -f deploy/backend.Dockerfile \
  -t <ID>/cac-backend:0.1.1 --push .

docker buildx build --platform linux/amd64 -f deploy/frontend.Dockerfile \
  -t <ID>/cac-frontend:0.1.0 --build-arg NEXT_PUBLIC_API_BASE="" --push frontend
```

backend는 저장소 루트, frontend는 `frontend/`가 빌드 컨텍스트입니다.
`NEXT_PUBLIC_API_BASE=""`는 게이트웨이 동일 오리진 배포를 전제로 한 값입니다.

## 참고: 다중 노드 공유 볼륨

AWS에서는 `efs-rag-assets.yaml`(EFS, RWX)이 이 역할을 합니다. kind에서 `kind/nfs-fallback.yaml`은
RWO PVC라 backend를 단일 노드에서만 스케줄할 수 있습니다. 여러 노드에서 RAG 자산을 공유해야 하면
NFS 프로비저너를 설치하고 `kind/nfs.yaml`(RWX)을 대신 적용합니다.

```bash
helm repo add nfs-ganesha https://kubernetes-sigs.github.io/nfs-ganesha-server-and-external-provisioner/
helm install nfs-server nfs-ganesha/nfs-server-provisioner -n cac -f deploy/k8s/kind/nfs-values.yaml
kubectl apply -f deploy/k8s/kind/nfs.yaml
```

## CI/CD (운영 클러스터)

사람이 결정하는 지점은 `develop → main` PR 머지 하나입니다. 그 뒤는 자동입니다.

1. main push → GitHub Actions(`.github/workflows/ci.yml`)가 검사(py_compile·eslint·tsc)
2. 변경된 쪽만 이미지 빌드 → OIDC로 AWS 역할 assume → ECR push (태그 `main-<sha7>`)
3. `deploy/k8s/kustomization.yaml`의 `newTag`를 바꾼 봇 커밋을 main에 push (`[skip ci]`)
4. 클러스터의 Argo CD가 main의 `deploy/k8s`를 보고 자동 동기화 → 롤링 업데이트

- PR(develop/main 대상)에서는 1번 검사만 돕니다.
- GitHub Secret은 `AWS_ROLE_ARN` 하나(`terraform output github_actions_role_arn`). 액세스 키 없음.
- 롤백: 봇 커밋을 `git revert` → main push.
- Argo CD 설치·UI(`https://argocd.cloudinfra.help`)·웹훅은 `deploy/argocd/README.md` 참고.

## 정리

```bash
kubectl delete namespace cac
helm uninstall eg -n envoy-gateway-system
```
