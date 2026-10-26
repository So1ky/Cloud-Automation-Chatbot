import os
import subprocess
import tempfile
import logging
import base64
from fastapi import HTTPException

# 로깅 설정 (에러 확인용)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_diagram(diagram_yaml: str) -> str:
    # 1. 임시 YAML 파일 생성
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
    ) as temp_yaml:
        temp_yaml.write(diagram_yaml)
        temp_yaml_path = temp_yaml.name
    
    # 2. 임시 PNG 이미지 저장 경로 설정
    temp_img_path = temp_yaml_path + ".png"
    
    logger.info("Generating diagram with awsdac...")
    
    try:
        # 3. awsdac로 임시 이미지 생성
        subprocess.run(
            ["awsdac", temp_yaml_path, "-o", temp_img_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        logger.info("awsdac execution successful")
        
        # 4. 이미지 파일을 읽어 Base64 텍스트로 인코딩
        with open(temp_img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        
        return f"data:image/png;base64,{encoded_string}"
        
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
    finally:
        # 5. 임시 생성된 파일들을 디스크에서 즉시 지웁니다.
        if os.path.exists(temp_yaml_path):
            try:
                os.remove(temp_yaml_path)
            except Exception as e:
                logger.error(f"Failed to remove temp yaml: {e}")
        if os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except Exception as e:
                logger.error(f"Failed to remove temp image: {e}")
