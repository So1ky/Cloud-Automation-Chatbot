"""진입점: 설계 에이전트 실행 (사용자 자연어 입력 모드)."""

import sys
import io

# 한글 입출력 인코딩 보장
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")


from dotenv import load_dotenv

load_dotenv()

from ai_engine.graph import run_design_agent


def main() -> None:
    print("=" * 55)
    print("   Cloud Infrastructure Design Agent")
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
        print("설계 중...\n")

        try:
            result = run_design_agent(user_input)

            print("\n[출력] 생성된 YAML 아키텍처 명세:")
            print("-" * 55)
            print(result["yaml_output"])
            print("-" * 55)
            print("\n다음 요구사항을 입력하거나 'exit'으로 종료하세요.\n")

        except Exception as e:
            print(f"[오류] {e}\n")


if __name__ == "__main__":
    main()
