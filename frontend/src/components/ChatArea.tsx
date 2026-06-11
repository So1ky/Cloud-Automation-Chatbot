"use client";
import React, { useRef, useEffect } from "react";

interface Message {
  id: string;
  role: "user" | "bot";
  text: string;
  imageUrl?: string;
  timestamp: Date;
}

interface ChatAreaProps {
  messages: Message[];
  isLoading: boolean;
  isHistoryLoading: boolean;
  healthStatus: "ok" | "error" | "loading";
  inputValue: string;
  setInputValue: (val: string) => void;
  onSend: () => void;
  onStartNewChat: () => void;
  onOpenAuth: () => void;
  currentUser: { id: number; email: string } | null;
}

export default function ChatArea({
  messages,
  isLoading,
  isHistoryLoading,
  healthStatus,
  inputValue,
  setInputValue,
  onSend,
  onStartNewChat,
  onOpenAuth,
  currentUser,
}: ChatAreaProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <section className="flex min-w-0 flex-1 flex-col">
      {/* Main Header */}
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-4 shadow-sm">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-bold text-slate-800 tracking-tight">Cloud Architecture Assistant</h1>
          <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold ${
            healthStatus === "ok" ? "bg-green-50 text-green-700" : "bg-red-50 text-red-700"
          }`}>
            <span className={`h-1.5 w-1.5 rounded-full ${healthStatus === "ok" ? "bg-green-500" : "bg-red-500"}`} />
            {healthStatus === "ok" ? "Backend Online" : "Connection Error"}
          </span>
        </div>

        <div className="flex items-center gap-3">
          {!currentUser && (
            <button
              onClick={onOpenAuth}
              className="rounded-xl bg-blue-600 px-4 py-2 text-xs font-bold text-white transition-all hover:bg-blue-700 active:scale-95 shadow-md"
            >
              로그인 / 회원가입
            </button>
          )}
        </div>
      </header>

      {/* Chat History Messages Scroll area */}
      <div className="flex-1 overflow-y-auto bg-slate-50/50 p-6">
        <div className="mx-auto max-w-4xl space-y-6">
          {isHistoryLoading ? (
            <div className="flex flex-col items-center justify-center py-20 text-slate-400">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-slate-300 border-t-blue-600 mb-3" />
              <span className="text-sm font-medium">채팅 내역을 가져오는 중입니다...</span>
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
                  className={`max-w-[85%] rounded-2xl px-5 py-4 shadow-sm leading-relaxed transition-all ${
                    msg.role === "user"
                      ? "rounded-tr-none bg-blue-600 text-white font-medium"
                      : "rounded-tl-none border border-slate-200 bg-white text-slate-800"
                  }`}
                >
                  <p className="whitespace-pre-wrap text-[15px]">{msg.text}</p>
                  {msg.imageUrl && (
                    <div className="mt-4 overflow-hidden rounded-xl border border-slate-200 bg-slate-50 shadow-inner">
                      <img
                        src={msg.imageUrl}
                        alt="Generated Diagram"
                        className="h-auto max-h-[500px] w-full object-contain hover:scale-[1.01] transition-transform"
                      />
                      <div className="border-t border-slate-200 bg-white p-3 text-center text-xs">
                        <button
                          onClick={() => {
                            try {
                              // Base64 문자열을 바이너리 데이터로 디코딩
                              const base64Data = msg.imageUrl!.split(",")[1];
                              const byteCharacters = atob(base64Data);
                              const byteNumbers = new Array(byteCharacters.length);
                              for (let i = 0; i < byteCharacters.length; i++) {
                                byteNumbers[i] = byteCharacters.charCodeAt(i);
                              }
                              const byteArray = new Uint8Array(byteNumbers);
                              
                              // 바이너리 데이터를 Blob 객체로 생성 (MIME 타입 지정)
                              const blob = new Blob([byteArray], { type: "image/png" });
                              
                              // Blob을 가리키는 고유 Object URL 생성
                              const blobUrl = URL.createObjectURL(blob);
                              
                              // 새 탭에서 Blob URL 열기
                              window.open(blobUrl, "_blank");
                            } catch (error) {
                              console.error("Failed to open Base64 image in new tab:", error);
                            }
                          }}
                          className="inline-flex items-center gap-1 font-semibold text-blue-600 hover:underline cursor-pointer"
                        >
                          <span>새 탭에서 원본 이미지 열기</span>
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
                            <path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z" />
                            <path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  )}
                  <span
                    className={`mt-2 block text-[10px] tracking-wide uppercase opacity-70 ${
                      msg.role === "user" ? "text-right text-blue-100" : "text-left text-slate-400"
                    }`}
                  >
                    {new Date(msg.timestamp).toLocaleTimeString([], {
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
              <div className="rounded-2xl rounded-tl-none border border-slate-200 bg-white px-5 py-4 shadow-sm">
                <div className="flex items-center gap-3">
                  <span className="text-xs text-slate-400 font-semibold animate-pulse">아키텍처 설계 중...</span>
                  <div className="flex gap-1">
                    <div className="h-2 w-2 animate-bounce rounded-full bg-blue-600"></div>
                    <div className="h-2 w-2 animate-bounce rounded-full bg-blue-600 [animation-delay:-.3s]"></div>
                    <div className="h-2 w-2 animate-bounce rounded-full bg-blue-600 [animation-delay:-.5s]"></div>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Footer Area */}
      {!messages.some((msg) => msg.role === "user") ? (
        <div className="border-t border-slate-200 bg-white p-5">
          <div className="mx-auto flex max-w-4xl gap-4 items-center">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onSend()}
              placeholder="예: 3티어 웹 어플리케이션 아키텍처 그려줘"
              className="flex-1 rounded-xl border border-slate-200 px-4 py-3 text-[15px] shadow-sm outline-none transition-all focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 disabled:bg-slate-50 disabled:text-slate-400"
              disabled={isLoading}
            />
            <button
              onClick={onSend}
              disabled={isLoading || !inputValue.trim()}
              className="rounded-xl bg-blue-600 px-6 py-3 font-semibold text-white shadow-md transition-all hover:bg-blue-700 disabled:bg-slate-300 disabled:shadow-none hover:shadow-lg active:scale-95 disabled:active:scale-100"
            >
              다이어그램 생성
            </button>
          </div>
        </div>
      ) : null}
    </section>
  );
}
