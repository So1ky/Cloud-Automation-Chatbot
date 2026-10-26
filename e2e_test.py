"""전체 파이프라인 e2e 테스트: 설계 → 개발 → 검증 (+Self-Healing).

사용법:
    python3 e2e_test.py                       # 기본 요구사항(쇼핑몰) / 결과는 e2e_output/case1
    python3 e2e_test.py "<요구사항>" <케이스명>

결과물(YAML, tf 4개, validation_report.json)은 e2e_output/<케이스명>/ 에 저장된다.
"""

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))
load_dotenv(PROJECT / ".env")

OUT_DIR = PROJECT / "e2e_output"
OUT_DIR.mkdir(exist_ok=True)

from ai_engine.graph import run_pipeline  # noqa: E402

REQUIREMENT = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else (
    "React 프론트엔드와 Node.js 백엔드로 구성된 쇼핑몰 웹 서비스를 만들고 싶어. "
    "백엔드는 ECS로 돌리고 데이터베이스는 RDS를 쓸 거야. "
    "트래픽이 몰릴 때를 대비해 오토스케일링이 필요하고, 회원 로그인 기능도 있어."
)
CASE = sys.argv[2] if len(sys.argv) > 2 else "case1"

print(f"### 요구사항: {REQUIREMENT}\n", flush=True)
start = time.time()
result = run_pipeline(REQUIREMENT)
elapsed = time.time() - start

case_dir = OUT_DIR / CASE
case_dir.mkdir(exist_ok=True)
(case_dir / "architecture.yaml").write_text(result["yaml_output"], encoding="utf-8")
for name, content in result["terraform_files"].items():
    (case_dir / name).write_text(content, encoding="utf-8")
(case_dir / "validation_report.json").write_text(
    json.dumps(result["validation_report"], ensure_ascii=False, indent=2), encoding="utf-8"
)
if result.get("validation_summary"):
    (case_dir / "user_summary.md").write_text(result["validation_summary"], encoding="utf-8")

report = result["validation_report"]
print("\n" + "=" * 60)
print(f"### 결과 요약 (소요 {elapsed:.0f}초)")
print(f"- 최종 판정: {'통과' if result['validation_passed'] else '실패'}")
print(f"- 검증 실행: {report.get('attempts', 1)}회")
syntax = report.get("terraform_validate", {})
print(f"- terraform validate: skipped={syntax.get('skipped')} passed={syntax.get('passed')} "
      f"errors={len(syntax.get('errors', []))}")
for section, label in (("security_analysis", "보안 분석"), ("optimization_analysis", "최적화 분석")):
    data = report.get(section, {})
    if data.get("skipped"):
        print(f"- {label}: 건너뜀 ({data.get('reason')})")
    else:
        print(f"- {label}: 이슈 {len(data.get('issues', []))}개")
        for i in data.get("issues", []):
            print(f"    [{i['severity']}/{i['category']}] {i['description'][:100]}")
cost = report.get("cost_estimate", {})
if cost.get("skipped"):
    print(f"- 비용 분석: 건너뜀 ({cost.get('reason')})")
else:
    print(f"- 비용 분석: 월 예상 {cost.get('total_monthly_cost')} {cost.get('currency')}")
print(f"- 산출물 저장: {case_dir}")
