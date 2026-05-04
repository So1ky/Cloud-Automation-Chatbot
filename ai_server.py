"""FastAPI 서버: AI 설계 에이전트를 HTTP API로 제공한다.

엔드포인트:
    POST /design  - 사용자 요구사항 → YAML 아키텍처 명세
    GET  /health  - 서버 상태 확인
"""

from contextlib import asynccontextmanager

import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from ai_engine.graph import run_design_agent
from ai_engine.rag.knowledge_base import load_knowledge_base
from ai_engine.agents.design.converter import convert_to_diagram_yaml


# ─── 시작 시 ChromaDB 미리 로드 (첫 요청 지연 방지) ─────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[서버] ChromaDB 워밍업 중...")
    load_knowledge_base()
    print("[서버] 준비 완료")
    yield


# ─── 앱 설정 ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cloud Infrastructure Design Agent API",
    description="자연어 요구사항을 AWS 아키텍처 YAML 명세로 변환합니다.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 단계: 모든 출처 허용 (배포 시 프론트 도메인으로 교체)
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 요청/응답 스키마 ─────────────────────────────────────────────────────────

class DesignRequest(BaseModel):
    requirements: str


class DesignResponse(BaseModel):
    yaml_output: str   # 우리 YAML 명세 (Terraform 생성용)
    diagram_yaml: str  # awsdac 전용 YAML (다이어그램 생성용)


# ─── 엔드포인트 ──────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/design", response_model=DesignResponse)
def design(req: DesignRequest):
    """사용자 자연어 요구사항을 받아 AWS 아키텍처 YAML 명세를 반환한다."""
    if not req.requirements.strip():
        raise HTTPException(status_code=400, detail="requirements가 비어 있습니다.")

    try:
        result = run_design_agent(req.requirements)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"에이전트 실행 오류: {e}")

    yaml_output = result["yaml_output"]

    # awsdac 다이어그램 YAML 변환
    try:
        parsed = yaml.safe_load(yaml_output)
        arch_data = parsed.get("architecture", {}) if parsed else {}
        diagram_dict = convert_to_diagram_yaml(arch_data)
        diagram_yaml = yaml.dump(diagram_dict, allow_unicode=True, sort_keys=False, default_flow_style=False)
    except Exception:
        diagram_yaml = ""

    return DesignResponse(yaml_output=yaml_output, diagram_yaml=diagram_yaml)


# ─── 로컬 실행 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("ai_server:app", host="0.0.0.0", port=8000, reload=False)
