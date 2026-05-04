import yaml
import logging
from fastapi import HTTPException

from ai_engine.graph import run_design_agent
from ai_engine.agents.design.converter import convert_to_diagram_yaml
from backend.schema.design_schema import DesignRequest, DesignResponse

# 로깅 설정 (에러 확인용)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def design(req: DesignRequest) -> DesignResponse:
    if not req.requirements.strip():
        raise HTTPException(status_code=400, detail="requirements가 비어 있습니다.")

    try:
        result = run_design_agent(req.requirements)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"에이전트 실행 오류: {e}")

    yaml_output = result["yaml_output"]

    try:
        parsed = yaml.safe_load(yaml_output)
        arch_data = parsed.get("architecture", {}) if parsed else {}
        diagram_dict = convert_to_diagram_yaml(arch_data)
        diagram_yaml = yaml.dump(
            diagram_dict,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
    except Exception as e:
        logger.error(f"diagram yaml conversion failed: {e}")
        raise HTTPException(status_code=500, detail="다이어그램 YAML 변환 실패")

    return DesignResponse(yaml_output=yaml_output, diagram_yaml=diagram_yaml)
