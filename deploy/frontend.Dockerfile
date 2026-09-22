# frontend 이미지 (빌드 컨텍스트: frontend/)
# 빌드: docker build -f deploy/frontend.Dockerfile -t cac-frontend:0.1.0 --build-arg NEXT_PUBLIC_API_BASE="" frontend

# ── 의존성 + 빌드 스테이지 ─────────────────────────────────────
FROM node:22-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# NEXT_PUBLIC_*는 빌드 타임에 인라인됨. 배포는 ""(동일 오리진), 기본값은 로컬 dev용
ARG NEXT_PUBLIC_API_BASE=""
ENV NEXT_PUBLIC_API_BASE=${NEXT_PUBLIC_API_BASE}
RUN npm run build

# ── 런타임 스테이지 (standalone 최소 번들) ─────────────────────
FROM node:22-alpine

WORKDIR /app
ENV NODE_ENV=production \
    HOSTNAME=0.0.0.0 \
    PORT=3000

COPY --from=builder --chown=node:node /app/.next/standalone ./
COPY --from=builder --chown=node:node /app/.next/static ./.next/static
COPY --from=builder --chown=node:node /app/public ./public

USER node
EXPOSE 3000
CMD ["node", "server.js"]
