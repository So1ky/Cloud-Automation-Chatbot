# 모니터링·로깅·알림 (control-plane에서, Phase별 1회)

설계: `docs/superpowers/specs/2026-10-27-monitoring-design.md`(로컬). 스택: kube-prometheus-stack 92.14.0 / loki 7.5.0 / alloy 1.14.3 / metrics-server 3.14.0.
전부 Argo CD Application(`deploy/argocd/*.yaml`)으로 배포한다. 차트 values는 `deploy/monitoring/*-values.yaml`,
우리 관측 대상(CR·대시보드)은 `deploy/monitoring/config/`.

공통 규칙: Application은 **먼저 수동 모드로 등록 → UI에서 diff 확인 → Sync → 자동 전환**.
수동 모드 = 파일에서 `automated:` 블록만 뗀 것:

```bash
cd ~/cac-git && git pull
sed '/^    automated:/,$d' deploy/argocd/<name>.yaml | kubectl apply -f -   # 수동 등록
# UI에서 diff 확인 → SYNC
kubectl apply -f deploy/argocd/<name>.yaml                                  # 자동 전환
kubectl -n argocd get application <name> -o jsonpath='{.status.sync.status} {.status.health.status}{"\n"}'
```

## 0. 선행 조건

- main에 `https-grafana` 리스너가 머지되어 `cac` 앱이 Synced, `kubectl -n cac get certificate grafana-cloudinfra-help-tls` READY=True
- `terraform apply`로 `grafana.cloudinfra.help` CNAME 생성됨
- 시크릿:
```bash
kubectl create namespace monitoring
cp deploy/monitoring/secrets.example.yaml deploy/monitoring/secrets.yaml   # 값 채우기 (Discord 웹훅, Grafana 비밀번호)
kubectl apply -f deploy/monitoring/secrets.yaml
```

`secrets.yaml`은 `.gitignore`로 제외되어 있으며 절대 커밋하지 말 것. 웹훅 URL이 유출되면 Discord에서 재발급한다.
Grafana admin 비밀번호는 **최초 초기화 때만** `grafana-admin` Secret에서 읽어 DB(PVC)에 저장된다. 나중에 Secret을 바꿔도 비밀번호는 바뀌지 않으므로, 변경은 Grafana UI(또는 `grafana cli admin reset-admin-password`)로 한다.

## 1. Phase 1 — 메트릭

등록 순서: `metrics-server` → `kube-prometheus-stack`(Healthy까지 2~3분, CRD 생성) → `monitoring-config`.

`monitoring-config`는 ServiceMonitor/PodMonitor/PrometheusRule/AlertmanagerConfig CR을 담고 있고 이 CRD는 `kube-prometheus-stack`이 만든다. 따라서 그 Application이 Healthy가 된 뒤에만 등록한다. 먼저 등록하면 Sync가 "no matches for kind"로 실패한다.

확인:
```bash
kubectl top nodes && kubectl top pods -n cac
kubectl -n monitoring get pods -o wide            # stateful 파드 전부 control-plane(ip-10-0-1-10), node-exporter 3개
kubectl -n monitoring get pvc                      # prometheus 10Gi, grafana 1Gi Bound
free -m | awk 'NR==2{print "available MB:",$7}'   # 10분 뒤 ≥ 500
```
브라우저 `https://grafana.cloudinfra.help` → admin / secrets.yaml 비밀번호. Dashboards → "CAC / 앱 개요", "CAC / Envoy RED".
Prometheus 타깃 확인(control-plane에서 exec, 기대값 `DOWN: 0`):
```bash
kubectl -n monitoring exec sts/prometheus-kube-prometheus-stack-prometheus -c prometheus -- \
  wget -qO- 'http://localhost:9090/api/v1/targets?state=active' | python3 -c \
  "import json,sys; t=json.load(sys.stdin)['data']['activeTargets']; print({x['scrapePool']: x['health'] for x in t}); print('DOWN:', sum(x['health']!='up' for x in t))"
```
Argo 상태 확인: `kubectl -n argocd get applications` — metrics-server, kube-prometheus-stack, monitoring-config 모두 `Synced` / `Healthy`.

## 2. Phase 2 — 로그

등록 순서: `loki`(Healthy) → `alloy`. 확인: `kubectl -n monitoring logs ds/alloy --tail=20`에 에러 없음,
Grafana Explore → Loki → `{namespace="cac", container="backend"}`.
`kubectl -n argocd get applications`에 loki, alloy가 추가로 `Synced` / `Healthy`로 보여야 한다.

## 3. Phase 3 — 알림 테스트

`deploy/monitoring/config/prometheusrule.yaml`의 `cac.platform` 그룹에 아래 규칙을 **임시** 추가해 push(main 머지) → Argo Sync →
약 1분 내 Discord 메시지 → 규칙 제거 push → resolved 메시지.
```yaml
        - alert: AlwaysFiring
          expr: vector(1)
          labels:
            severity: critical
          annotations:
            summary: "알림 경로 테스트 — 보이면 제거할 것"
```

## 운영 메모

- 클러스터를 꺼두면 모니터링도 꺼진다. PVC는 남아 cac-up 후 그래프가 이어진다.
- 가짜 알림이 보이면 `kube-prometheus-stack-values.yaml`의 `defaultRules.disabled`에 이름 추가 → PR.
- Prometheus·Alertmanager·Loki UI는 비공개. 상태 확인은 위처럼 `kubectl exec` + wget.
  브라우저가 필요하면 노트북에서 `aws ssm start-session --target <control-plane 인스턴스 ID> --document-name AWS-StartPortForwardingSession --parameters '{"portNumber":["9090"],"localPortNumber":["9090"]}'` 로 포워딩한 뒤,
  control-plane에서 `kubectl -n monitoring port-forward --address 0.0.0.0 svc/kube-prometheus-stack-prometheus 9090`.
- 메모리가 1.3GB를 넘으면 values에서 `retention: 5d`, `scrapeInterval: 60s`로 하향.
