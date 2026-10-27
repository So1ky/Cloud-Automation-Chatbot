import { API_BASE, ApiError } from "./api";

// SSE 이벤트와 UTF-8 문자가 네트워크 청크 사이에서 나뉘어 도착할 수 있다.
export async function consumeSSE(
  body: ReadableStream<Uint8Array>,
  onEvent: (event: string, data: Record<string, unknown>) => void,
) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      // 청크 사이에서 나뉜 경우도 포함하여 완성된 CRLF 쌍만 LF로 변환한다.
      buffer = buffer.replace(/\r\n/g, "\n");
      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        let event = "message";
        const data: string[] = [];
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trimStart();
          if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
        }
        if (data.length) onEvent(event, JSON.parse(data.join("\n")));
      }
      if (done) return;
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export async function streamChat(
  request: { requirements: string; conversation_id: number | null },
  onEvent: (event: string, data: Record<string, unknown>) => void,
  signal?: AbortSignal,
) {
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal?.addEventListener("abort", abort, { once: true });
  if (signal?.aborted) controller.abort();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, 600000);
  let completed = false;
  try {
    const response = await fetch(`${API_BASE}/api/chat/stream`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(request),
      signal: controller.signal,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      const detail = typeof payload?.detail === "string" ? payload.detail : null;
      throw new ApiError("http", detail || "요청이 실패했습니다.", response.status, detail);
    }
    if (!response.body || !response.headers.get("content-type")?.includes("text/event-stream")) {
      throw new Error("진행 상황 응답을 받을 수 없습니다.");
    }
    await consumeSSE(response.body, (event, data) => {
      if (event === "error") {
        throw new Error(String(data.message || "생성 중 오류가 발생했습니다."));
      }
      if (event === "done") completed = true;
      onEvent(event, data);
    });
    if (!completed) throw new Error("완료 전에 연결이 끊겼습니다. 대화 내역을 확인해 주세요.");
  } catch (error) {
    if (timedOut) throw new ApiError("timeout", "요청 시간이 초과되었습니다. 대화 내역을 확인해 주세요.");
    if (error instanceof TypeError) throw new ApiError("network", "서버 연결이 끊겼습니다. 대화 내역을 확인해 주세요.");
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", abort);
  }
}
