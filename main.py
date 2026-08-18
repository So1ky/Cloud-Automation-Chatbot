"""진입점: 설계 + 개발 에이전트 실행 (사용자 자연어 입력 모드)."""

import sys
import io

# 한글 입출력 인코딩 보장
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")


from dotenv import load_dotenv

load_dotenv()

from ai_engine.graph import run_pipeline


def main() -> None:
    print("=" * 55)
    print("   Cloud Infrastructure Design & IaC Agent")
    print("=" * 55)
    print("AWS 클라우드 인프라 요구사항을 자연어로 입력하세요.")
    print("(종료: 'exit' 또는 'quit' 입력)\n")

    while True:
        try:
            user_input = input("요구사항 입력 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("종료합니다.")
            break

        print(f"\n[입력] {user_input}\n")
        print("설계 및 코드 생성 중...\n")

        try:
            result = run_pipeline(user_input)

            print("\n[1단계] 생성된 YAML 아키텍처 명세:")
            print("-" * 55)
            print(result["yaml_output"])
            print("-" * 55)

            print("\n[2단계] 생성된 Terraform 파일:")
            for filename, content in result["terraform_files"].items():
                print(f"\n{'=' * 55}")
                print(f"  {filename}")
                print("=" * 55)
                print(content)

            print("\n[3단계] 검증 결과:")
            print("-" * 55)
            report = result.get("validation_report", {})
            print(f"최종 판정: {'통과' if result.get('validation_passed') else '실패'} "
                  f"(검증 시도 {report.get('attempts', 1)}회)")

            syntax = report.get("terraform_validate", {})
            if syntax.get("skipped"):
                print(f"- terraform validate: 건너뜀 ({syntax.get('reason')})")
            else:
                print(f"- terraform validate: {'통과' if syntax.get('passed') else '실패'}")
                for err in syntax.get("errors", []):
                    print(f"    오류: {err['summary']}")

            security = report.get("security_analysis", {})
            if security.get("skipped"):
                print(f"- 보안/모범사례 분석: 건너뜀 ({security.get('reason')})")
            else:
                issues = security.get("issues", [])
                print(f"- 보안/모범사례 분석: 이슈 {len(issues)}개")
                for issue in issues:
                    print(f"    [{issue['severity']}/{issue['category']}] {issue['description']}")

            optimization = report.get("optimization_analysis", {})
            if optimization.get("skipped"):
                print(f"- 최적화 분석: 건너뜀 ({optimization.get('reason')})")
            else:
                opt_issues = optimization.get("issues", [])
                print(f"- 최적화 분석: 이슈 {len(opt_issues)}개")
                for issue in opt_issues:
                    print(f"    [{issue['severity']}/{issue['category']}] {issue['description']}")

            cost = report.get("cost_estimate", {})
            if cost.get("skipped"):
                print(f"- 비용 분석: 건너뜀 ({cost.get('reason')})")
            else:
                print(f"- 비용 분석: 월 예상 {cost.get('total_monthly_cost')} {cost.get('currency')}")

            if result.get("validation_summary"):
                print("\n[4단계] 사용자 안내문:")
                print("-" * 55)
                print(result["validation_summary"])

            print("\n다음 요구사항을 입력하거나 'exit'으로 종료하세요.\n")

        except Exception as e:
            print(f"[오류] {e}\n")


if __name__ == "__main__":
    main()
