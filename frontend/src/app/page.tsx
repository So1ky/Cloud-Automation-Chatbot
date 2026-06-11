"use client";
import { useEffect, useRef, useState } from "react";

interface Message {
  id: string;
  role: "user" | "bot";
  text: string;
  imageUrl?: string;
  timestamp: Date;
}

interface ChatHistoryItem {
  id: number;
  requirements: string;
  created_at: string;
}

interface ChatHistoryDetail extends ChatHistoryItem {
  response_message: string;
  image_url?: string | null;
}

const initialMessage = (): Message => ({
  id: "welcome",
  role: "bot",
  text: "안녕하세요! 어떤 클라우드 아키텍처를 그려드릴까요?",
  timestamp: new Date(),
});

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([initialMessage()]);
  const [chatHistory, setChatHistory] = useState<ChatHistoryItem[]>([]);
  const [selectedChatId, setSelectedChatId] = useState<number | null>(null);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [healthStatus, setHealthStatus] = useState<"ok" | "error" | "loading">(
    "loading",
  );
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const loadChatHistory = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/chat/history");
      if (!response.ok) throw new Error("채팅 목록 조회 실패");

      const history: ChatHistoryItem[] = await response.json();
      setChatHistory(history);
    } catch (error) {
      console.error("Failed to load chat history:", error);
    }
  };

  const loadChatDetail = async (chatId: number) => {
    setIsHistoryLoading(true);

    try {
      const response = await fetch(`http://localhost:8000/api/chat/${chatId}`);
      if (!response.ok) throw new Error("채팅 상세 조회 실패");

      const chat: ChatHistoryDetail = await response.json();
      const timestamp = new Date(chat.created_at);

      setSelectedChatId(chat.id);
      setMessages([
        {
          id: `${chat.id}-user`,
          role: "user",
          text: chat.requirements,
          timestamp,
        },
        {
          id: `${chat.id}-bot`,
          role: "bot",
          text: chat.response_message,
          imageUrl: chat.image_url || undefined,
          timestamp,
        },
      ]);
    } catch (error) {
      console.error("Failed to load chat detail:", error);
    } finally {
      setIsHistoryLoading(false);
    }
  };

  const startNewChat = () => {
    setSelectedChatId(null);
    setMessages([initialMessage()]);
    setInputValue("");
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    fetch("http://localhost:8000/api/health")
      .then((res) => res.json())
      .then(() => setHealthStatus("ok"))
      .catch(() => setHealthStatus("error"));

    loadChatHistory();
  }, []);

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      text: inputValue,
      timestamp: new Date(),
    };

    setSelectedChatId(null);
    setMessages([userMessage]);
    setInputValue("");
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ requirements: userMessage.text }),
      });

      if (!response.ok) throw new Error("서버 응답 오류");

      const data = await response.json();
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "bot",
        text: data.message || "생성이 완료되었습니다.",
        imageUrl: data.image_url,
        timestamp: new Date(),
      };

      setMessages([userMessage, botMessage]);
      setSelectedChatId(data.chat_id);
      await loadChatHistory();
    } catch (error) {
      console.error("Error:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "bot",
        text: "죄송합니다. 오류가 발생했습니다. 백엔드 서버가 실행 중인지 확인해 주세요.",
        timestamp: new Date(),
      };
      setMessages([userMessage, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex h-screen bg-gray-100">
      <aside className="hidden w-72 shrink-0 border-r bg-white md:flex md:flex-col">
        <div className="border-b p-4">
          <button
            onClick={startNewChat}
            className="w-full rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700"
          >
            새 채팅
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-3">
          {chatHistory.length === 0 ? (
            <p className="px-2 py-3 text-sm text-gray-500">
              저장된 채팅이 없습니다.
            </p>
          ) : (
            <div className="space-y-1">
              {chatHistory.map((chat) => (
                <button
                  key={chat.id}
                  onClick={() => loadChatDetail(chat.id)}
                  className={`w-full rounded-lg px-3 py-2 text-left transition-colors ${
                    selectedChatId === chat.id
                      ? "bg-blue-50 text-blue-700"
                      : "text-gray-700 hover:bg-gray-100"
                  }`}
                >
                  <span className="block truncate text-sm font-medium">
                    {chat.requirements}
                  </span>
                  <span className="mt-1 block text-xs text-gray-500">
                    {new Date(chat.created_at).toLocaleString([], {
                      month: "2-digit",
                      day: "2-digit",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </aside>

      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b bg-white px-6 py-4 shadow-sm">
          <h1 className="text-xl font-bold text-blue-600">Cloud Diagram Bot</h1>
          <div className="flex items-center gap-2">
            <span
              className={`h-3 w-3 rounded-full ${
                healthStatus === "ok" ? "bg-green-500" : "bg-red-500"
              }`}
            ></span>
            <span className="text-sm text-gray-600">
              {healthStatus === "ok"
                ? "Backend Connected"
                : "Backend Disconnected"}
            </span>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-4">
          <div className="mx-auto max-w-4xl space-y-6">
            {isHistoryLoading ? (
              <div className="text-center text-sm text-gray-500">
                채팅을 불러오는 중입니다.
              </div>
            ) : (
              messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${
                    msg.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl p-4 ${
                      msg.role === "user"
                        ? "rounded-tr-none bg-blue-600 text-white"
                        : "rounded-tl-none border bg-white text-gray-800 shadow-sm"
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{msg.text}</p>
                    {msg.imageUrl && (
                      <div className="mt-4 overflow-hidden rounded-lg border bg-gray-50">
                        <img
                          src={msg.imageUrl}
                          alt="Generated Diagram"
                          className="h-auto max-h-[500px] w-full object-contain"
                        />
                        <div className="border-t bg-white p-2 text-center text-xs text-gray-500">
                          <a
                            href={msg.imageUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="hover:underline"
                          >
                            이미지 크게 보기
                          </a>
                        </div>
                      </div>
                    )}
                    <span
                      className={`mt-1 block text-[10px] opacity-70 ${
                        msg.role === "user" ? "text-right" : "text-left"
                      }`}
                    >
                      {msg.timestamp.toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
                  </div>
                </div>
              ))
            )}

            {isLoading && (
              <div className="flex justify-start">
                <div className="rounded-2xl rounded-tl-none border bg-white p-4 shadow-sm">
                  <div className="flex gap-1">
                    <div className="h-2 w-2 animate-bounce rounded-full bg-gray-400"></div>
                    <div className="h-2 w-2 animate-bounce rounded-full bg-gray-400 [animation-delay:-.3s]"></div>
                    <div className="h-2 w-2 animate-bounce rounded-full bg-gray-400 [animation-delay:-.5s]"></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        <div className="border-t bg-white p-4">
          <div className="mx-auto flex max-w-4xl gap-3">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder="메시지를 입력하세요..."
              className="flex-1 rounded-xl border px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={isLoading}
            />
            <button
              onClick={handleSend}
              disabled={isLoading || !inputValue.trim()}
              className="rounded-xl bg-blue-600 px-6 py-2 font-medium text-white transition-colors hover:bg-blue-700 disabled:bg-gray-400"
            >
              전송
            </button>
          </div>
        </div>
      </section>
    </main>
  );
}
