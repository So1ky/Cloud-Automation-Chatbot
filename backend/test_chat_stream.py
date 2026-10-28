"""실제 모델 호출이나 영구 DB 없이 SSE 전송 동작을 검증한다."""
import importlib
import json
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


class ChatStreamTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        stub = types.ModuleType("backend.service.design_service")
        stub.design = lambda *args, **kwargs: None
        with patch.dict(sys.modules, {"backend.service.design_service": stub}), patch.dict(
            os.environ, {"SECRET_KEY": "transport-test-secret-not-for-production"}
        ):
            self.router = importlib.import_module("backend.router.chat_router")
        from backend.database.config import Base, get_db
        from backend.database.models import ChatHistory
        self.model = ChatHistory
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine)
        self.app = FastAPI()
        self.app.include_router(self.router.router)
        def db():
            with self.sessions() as session:
                yield session
        self.app.dependency_overrides[get_db] = db
        self.app.dependency_overrides[self.router.user_service.get_current_user_optional] = lambda: None

    async def asyncTearDown(self):
        self.engine.dispose()

    async def request(self, fail_image=False, fail_save=False, fail_design=None):
        def design(req, previous_yaml=None, on_progress=None):
            if fail_design:
                raise fail_design
            on_progress("design", "설계 중")
            on_progress("verify", "검증 중")
            return SimpleNamespace(terraform_files={"main.tf": "resource {}"}, validation_summary="검증 완료",
                                   cost_estimate=None, diagram_yaml="diagram", yaml_output="architecture: {}")
        def diagram(value):
            if fail_image:
                raise RuntimeError("image test failure")
            return "data:image/png;base64,test"
        def sessions():
            if fail_save:
                raise RuntimeError("save test failure")
            return self.sessions()
        with patch.object(self.router, "design", design), patch.object(self.router, "generate_diagram", diagram), patch.object(self.router, "SessionLocal", sessions):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test") as client:
                return await client.post("/api/chat/stream", json={"requirements": "test"})

    async def test_order_and_persistence(self):
        response = await self.request()
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("event: ai_result", response.text)
        self.assertNotIn("event: diagram\n", response.text)
        self.assertEqual(response.text.count("event: done"), 1)
        self.assertLess(response.text.index('"stage": "diagram"'), response.text.index("event: done"))
        payload = json.loads(response.text.split("event: done\ndata: ", 1)[1].strip())
        self.assertEqual(payload["image_url"], "data:image/png;base64,test")
        self.assertEqual(payload["terraform_code"], {"main.tf": "resource {}"})
        self.assertEqual(payload["validation_summary"], "검증 완료")
        with self.sessions() as session:
            chat = session.query(self.model).one()
            self.assertEqual(chat.conversation_id, chat.id)
            self.assertEqual(chat.terraform_code, {"main.tf": "resource {}"})

    async def test_image_failure_saves_ai_result(self):
        response = await self.request(fail_image=True)
        self.assertIn("event: done", response.text)
        self.assertNotIn("event: diagram\n", response.text)
        payload = json.loads(response.text.split("event: done\ndata: ", 1)[1].strip())
        self.assertIsNone(payload["image_url"])
        self.assertTrue(payload["warning"])
        self.assertEqual(payload["terraform_code"], {"main.tf": "resource {}"})
        with self.sessions() as session:
            self.assertIsNone(session.query(self.model).one().image_url)

    async def test_save_failure_does_not_send_results(self):
        response = await self.request(fail_save=True)
        self.assertNotIn("terraform_code", response.text)
        self.assertIn("event: error", response.text)
        self.assertNotIn("event: done", response.text)

    async def test_http_exception_keeps_status_and_detail(self):
        from fastapi import HTTPException
        response = await self.request(fail_design=HTTPException(status_code=502, detail="OpenAI API 연결 실패"))
        self.assertIn("event: error", response.text)
        payload = json.loads(response.text.split("event: error\ndata: ", 1)[1].strip())
        self.assertEqual(payload["status"], 502)
        self.assertEqual(payload["message"], "OpenAI API 연결 실패")

    async def test_validation_before_sse(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test") as client:
            response = await client.post("/api/chat/stream", json={"requirements": "  "})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
