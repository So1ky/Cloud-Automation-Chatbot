from __future__ import annotations

import importlib


def check_import(module_name: str) -> bool:
    """Try importing a module and print the result."""
    try:
        importlib.import_module(module_name)
        print(f"[OK] {module_name} import succeeded")
        return True
    except Exception as exc:  # pragma: no cover - simple runtime smoke test
        print(f"[FAIL] {module_name} import failed: {exc}")
        return False


def main() -> None:
    modules = ["chromadb", "langsmith", "ragas"]
    results = [check_import(name) for name in modules]

    if all(results):
        print("\nAll imports passed.")
    else:
        print("\nSome imports failed. Reinstall the failed packages.")


if __name__ == "__main__":
    main()
