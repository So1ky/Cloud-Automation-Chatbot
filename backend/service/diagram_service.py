import os
import subprocess
import tempfile
import logging
from uuid import uuid4

from fastapi import HTTPException

# 로깅 설정 (에러 확인용)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ROUTER_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(ROUTER_DIR)
STATIC_DIR = os.path.join(BACKEND_DIR, "static")

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

def generate_diagram(diagram_yaml: str) -> str:
    output_filename = f"diagram-{uuid4().hex}.png"
    output_path = os.path.join(STATIC_DIR, output_filename)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
    ) as temp_yaml:
        temp_yaml.write(diagram_yaml)
        temp_yaml_path = temp_yaml.name
    
    logger.info("Generating diagram with awsdac...")
    
    try:
        subprocess.run(
            ["awsdac", temp_yaml_path, "-o", output_path],
            check=True, # 명령어 실패 시 예외 발생
            capture_output=True, # 표준 출력과 표준 에러 캡처
            text=True,  # 텍스트 모드로 출력 캡처
            timeout=30  # 타임아웃 설정 (30초)
        )
        logger.info("awsdac execution successful")
        return output_filename
        
    except subprocess.TimeoutExpired:
        logger.error("awsdac execution timed out")
        raise HTTPException(status_code=504, detail="다이어그램 생성 시간이 초과되었습니다.")
    except subprocess.CalledProcessError as e:
        logger.error(f"awsdac failed: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"awsdac 실행 실패: {e.stderr}")
    except FileNotFoundError:
        logger.error("awsdac command not found")
        raise HTTPException(status_code=500, detail="awsdac 명령어를 찾을 수 없습니다. Go와 awsdac가 설치되어 있고 PATH에 등록되어 있는지 확인해 주세요.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
