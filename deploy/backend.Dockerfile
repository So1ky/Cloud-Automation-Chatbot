# backend + ai_engine 이미지 (빌드 컨텍스트: 저장소 루트)
# 빌드: docker build -f deploy/backend.Dockerfile -t cac-backend:0.1.0 .

# ── 외부 바이너리 다운로드 스테이지 ─────────────────────────────
FROM python:3.12-slim AS tools

ARG TARGETARCH
ARG AWSDAC_VERSION=v0.24
ARG TERRAFORM_VERSION=1.16.2

RUN apt-get update && apt-get install -y --no-install-recommends curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# awsdac: 다이어그램 렌더링에 필수 (없으면 챗 응답이 500)
RUN curl -fsSL -o /tmp/awsdac.zip \
      "https://github.com/awslabs/diagram-as-code/releases/download/${AWSDAC_VERSION}/awsdac-${AWSDAC_VERSION}_linux-${TARGETARCH}.zip" \
    && unzip /tmp/awsdac.zip -d /tmp/awsdac \
    && install -m 0755 "$(find /tmp/awsdac -type f -name awsdac | head -1)" /usr/local/bin/awsdac

# terraform: verify 단계의 validate용 (없어도 skip되지만 포함)
RUN curl -fsSL -o /tmp/terraform.zip \
      "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_${TARGETARCH}.zip" \
    && unzip /tmp/terraform.zip -d /tmp/tf \
    && install -m 0755 /tmp/tf/terraform /usr/local/bin/terraform

# infracost v2: 비용 분석용 — verify 코드가 v2 명령(scan/inspect)을 사용 (INFRACOST_API_KEY는 Secret으로 주입)
RUN curl -fsSL https://raw.githubusercontent.com/infracost/cli/master/scripts/install.sh | sh

# ── 런타임 스테이지 ────────────────────────────────────────────
FROM python:3.12-slim

# 모든 상대 경로(.env, alembic, SQLite 폴백)가 저장소 루트 기준이므로 WORKDIR 고정
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache

# torch는 CPU 전용 인덱스로 먼저 설치해 CUDA 휠 유입 차단 (이미지 수 GB 절감)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# 리랭커 모델을 이미지에 미리 굽기 (런타임 다운로드 제거 → 기동 시간 단축)
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)"

COPY --from=tools /usr/local/bin/awsdac /usr/local/bin/terraform /usr/local/bin/infracost /usr/local/bin/

COPY backend/ backend/
COPY ai_engine/ ai_engine/
COPY migrations/ migrations/
COPY alembic.ini .

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
