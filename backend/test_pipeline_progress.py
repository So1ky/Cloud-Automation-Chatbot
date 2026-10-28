"""결과가 일정한 대체 에이전트로 실제 LangGraph의 실행 경로를 검증한다."""
import importlib.util
import sys
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


class PipelineProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 임시 모듈 교체로 재초기화되지 않도록 네이티브 의존성을 미리 불러온다.
        import langgraph.graph
        import langsmith.run_helpers

        def design(state):
            return {"yaml_output": state["user_requirements"], "rag_context": "context"}
        def develop(state):
            return {"terraform_files": {"main.tf": state["yaml_output"]}}
        def verify(state):
            retry = state.get("retry_count", 0)
            return {"validation_passed": retry > 0, "retry_count": retry + 1,
                    "fix_target": "develop", "validation_report": {}}
        modules = {}
        for name, attr, function in [
            ("ai_engine.agents.design.agent", "design_node", design),
            ("ai_engine.agents.develop.agent", "develop_node", develop),
            ("ai_engine.agents.verify.agent", "verify_node", verify),
            ("ai_engine.report", "generate_user_summary", lambda *args, **kwargs: "summary"),
        ]:
            module = types.ModuleType(name)
            setattr(module, attr, function)
            modules[name] = module
        cls.modules = modules
        spec = importlib.util.spec_from_file_location("progress_graph_test", Path(__file__).parents[1] / "ai_engine" / "graph.py")
        cls.graph = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(cls.graph)

    def test_retry_stage_order_and_final_result(self):
        events = []
        with patch.dict(sys.modules, self.modules):
            result = self.graph.run_pipeline("request", on_progress=lambda stage, message: events.append((stage, message)))
        self.assertEqual([stage for stage, _ in events], ["design", "develop", "verify", "develop", "verify", "summary"])
        self.assertIn("수정 1회차", events[3][1])
        self.assertEqual(result["terraform_files"], {"main.tf": "request"})
        self.assertEqual(result["validation_summary"], "summary")

    def test_optional_callback_preserves_legacy_calls(self):
        with patch.dict(sys.modules, self.modules):
            self.assertTrue(self.graph.run_pipeline("legacy")["validation_passed"])

    def test_callback_exception_does_not_break_pipeline(self):
        def broken(stage, message):
            raise RuntimeError("callback failure")
        with patch.dict(sys.modules, self.modules), self.assertLogs(self.graph.logger, level="WARNING") as logs:
            result = self.graph.run_pipeline("request", on_progress=broken)
        self.assertEqual(result["terraform_files"], {"main.tf": "request"})
        self.assertTrue(result["validation_passed"])
        self.assertEqual(len(logs.records), 6)  # design, develop, verify, develop, verify, summary

    def test_concurrent_requests_keep_callbacks_separate(self):
        def run(name):
            events = []
            result = self.graph.run_pipeline(name, on_progress=lambda stage, message: events.append(stage))
            return result, events
        with patch.dict(sys.modules, self.modules), ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run, ["first", "second"]))
        for name, (result, events) in zip(["first", "second"], results):
            self.assertEqual(result["terraform_files"]["main.tf"], name)
            self.assertEqual(events, ["design", "develop", "verify", "develop", "verify", "summary"])


if __name__ == "__main__":
    unittest.main()
