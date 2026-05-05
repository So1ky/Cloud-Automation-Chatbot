from pydantic import BaseModel

# design 요청/응답 스키마
class DesignRequest(BaseModel):
    requirements: str

class DesignResponse(BaseModel):
    yaml_output: str   # 우리 YAML 명세 (Terraform 생성용)
    diagram_yaml: str  # awsdac 전용 YAML (다이어그램 생성용)
