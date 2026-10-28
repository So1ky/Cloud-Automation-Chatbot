"""에이전트별 추론 강도 환경변수 우선순위를 검증한다 (LLM 호출 없음)."""
import os
import unittest
from unittest.mock import patch

from ai_engine.config import get_reasoning_effort


class ReasoningEffortTests(unittest.TestCase):
    def test_default_is_low(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_reasoning_effort("design"), "low")

    def test_common_value_applies_to_all_agents(self):
        with patch.dict(os.environ, {"AI_REASONING_EFFORT": "medium"}, clear=True):
            self.assertEqual(get_reasoning_effort("design"), "medium")
            self.assertEqual(get_reasoning_effort("develop"), "medium")

    def test_agent_specific_overrides_common(self):
        env = {"AI_REASONING_EFFORT": "low", "AI_REASONING_EFFORT_DESIGN": "high"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_reasoning_effort("design"), "high")
            self.assertEqual(get_reasoning_effort("develop"), "low")


if __name__ == "__main__":
    unittest.main()
