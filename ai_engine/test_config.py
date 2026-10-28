"""에이전트별 추론 강도 환경변수 우선순위를 검증한다 (LLM 호출 없음)."""
import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from ai_engine.config import ClaudeChat, get_llm, get_reasoning_effort


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


class _Pick(BaseModel):
    instance_type: str


class ClaudeProviderTests(unittest.TestCase):
    """claude-* 접두사 분기와 구조화 출력 래퍼 (LLM 호출 없음)."""

    def test_claude_prefix_selects_anthropic_with_effort(self):
        env = {"AI_MODEL_DESIGN": "claude-opus-5-5", "AI_REASONING_EFFORT_DESIGN": "high",
               "ANTHROPIC_API_KEY": "test"}
        with patch.dict(os.environ, env, clear=True):
            llm = get_llm("design")
        self.assertIsInstance(llm, ClaudeChat)
        self.assertEqual(llm.model_kwargs["output_config"]["effort"], "high")

    def test_effort_none_maps_to_low_for_claude(self):
        env = {"AI_MODEL_RAG": "claude-sonnet-5-5", "AI_REASONING_EFFORT": "none",
               "ANTHROPIC_API_KEY": "test"}
        with patch.dict(os.environ, env, clear=True):
            llm = get_llm("rag")
        self.assertEqual(llm.model_kwargs["output_config"]["effort"], "low")

    def test_structured_output_parses_tool_call_and_injects_instruction(self):
        llm = ClaudeChat(model="claude-opus-5-5", anthropic_api_key="test")
        seen = {}

        def fake_invoke(messages, *args, **kwargs):
            seen["messages"] = messages
            return AIMessage(content="", tool_calls=[
                {"name": "_Pick", "args": {"instance_type": "t3.micro"}, "id": "1"}])

        chain = llm.with_structured_output(_Pick)
        bound = chain.middle[0]
        with patch.object(type(bound), "invoke", side_effect=fake_invoke):
            result = chain.invoke([SystemMessage(content="sys"), HumanMessage(content="hi")])
        self.assertEqual(result, _Pick(instance_type="t3.micro"))
        # 지시 system 메시지는 선두 system 바로 뒤, Human 앞에 들어간다
        self.assertIsInstance(seen["messages"][1], SystemMessage)
        self.assertIn("_Pick", seen["messages"][1].content)
        self.assertIsInstance(seen["messages"][2], HumanMessage)

    def test_structured_output_decodes_stringified_nested_object(self):
        """Claude가 중첩 객체를 JSON 문자열로 넣어도 파싱된다 (video 케이스에서 실제 발생)."""
        class Spec(BaseModel):
            architecture: _Pick

        llm = ClaudeChat(model="claude-opus-5-5", anthropic_api_key="test")
        chain = llm.with_structured_output(Spec)
        bound = chain.middle[0]
        msg = AIMessage(content="", tool_calls=[
            {"name": "Spec", "args": {"architecture": '{"instance_type": "t3.\nmicro"}'}, "id": "1"}])
        with patch.object(type(bound), "invoke", return_value=msg):
            result = chain.invoke("hi")
        self.assertEqual(result.architecture.instance_type, "t3.\nmicro")  # 제어문자 포함도 복원

    def test_structured_output_tolerates_extra_closing_brace(self):
        """video_opus2에서 실제 발생: 문자열 끝에 닫는 중괄호가 하나 더 붙음."""
        class Spec(BaseModel):
            architecture: _Pick

        llm = ClaudeChat(model="claude-opus-5-5", anthropic_api_key="test")
        chain = llm.with_structured_output(Spec)
        bound = chain.middle[0]
        msg = AIMessage(content="", tool_calls=[
            {"name": "Spec", "args": {"architecture": '{"instance_type": "t3.micro"}}'}, "id": "1"}])
        with patch.object(type(bound), "invoke", return_value=msg):
            self.assertEqual(chain.invoke("hi").architecture.instance_type, "t3.micro")

    def test_structured_output_raises_without_tool_call(self):
        llm = ClaudeChat(model="claude-opus-5-5", anthropic_api_key="test")
        chain = llm.with_structured_output(_Pick)
        bound = chain.middle[0]
        with patch.object(type(bound), "invoke", return_value=AIMessage(content="plain text")):
            with self.assertRaises(RuntimeError):
                chain.invoke("hi")


if __name__ == "__main__":
    unittest.main()
