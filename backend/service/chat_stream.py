"""동기 작업 스레드와 SSE 응답을 요청별로 연결한다."""

import asyncio
import json
from threading import Event


class ChatStream:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.queue = asyncio.Queue()
        self.closed = Event()

    def emit(self, event: str, data: dict):
        if not self.closed.is_set():
            self.loop.call_soon_threadsafe(self._put, event, data)

    def _put(self, event, data):
        if not self.closed.is_set():
            self.queue.put_nowait((event, data))

    def progress(self, stage: str, message: str):
        self.emit("progress", {"stage": stage, "message": message})

    async def events(self):
        try:
            yield ": connected\n\n"
            while True:
                try:
                    event, data = await asyncio.wait_for(self.queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                if event in ("done", "error"):
                    return
        finally:
            self.closed.set()
