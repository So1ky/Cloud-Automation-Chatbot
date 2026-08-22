"use client";
import { useEffect, useRef, useState } from "react";
import Sidebar from "@/components/Sidebar";
import ChatArea from "@/components/ChatArea";
import AuthModal from "@/components/AuthModal";
import ToastContainer, { ToastItem, ToastType } from "@/components/Toast";
import { apiFetch, toUserMessage, ApiError } from "@/lib/api";

interface Message {
  id: string;
  role: "user" | "bot";
  text: string;
  imageUrl?: string;
  terraformCode?: Record<string, string>;
  validationSummary?: string;
  isError?: boolean;
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
  terraform_code?: Record<string, string> | null;
  validation_summary?: string | null;
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

  // Auth States
  const [currentUser, setCurrentUser] = useState<{
    id: number;
    email: string;
  } | null>(null);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);

  // Toast 알림
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const toastId = useRef(0);
  // 마지막 요구사항 (전송 실패 시 재시도용)
  const lastRequirement = useRef<string>("");

  const dismissToast = (id: number) =>
    setToasts((prev) => prev.filter((t) => t.id !== id));

  const pushToast = (message: string, type: ToastType = "error") => {
    const id = ++toastId.current;
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => dismissToast(id), 5000);
  };

  const fetchUserInfo = async () => {
    try {
      const response = await apiFetch("/api/user/me");
      const user = await response.json();
      setCurrentUser(user);
      loadChatHistory();
    } catch (error) {
      // 미로그인(401)은 정상 상황이므로 조용히 처리
      if (!(error instanceof ApiError && error.status === 401)) {
        console.error("Failed to fetch user info:", error);
      }
      setCurrentUser(null);
      setChatHistory([]);
    }
  };

  const loadChatHistory = async () => {
    try {
      const response = await apiFetch("/api/chat/history");
      const history: ChatHistoryItem[] = await response.json();
      setChatHistory(history);
    } catch (error) {
      console.error("Failed to load chat history:", error);
      pushToast(toUserMessage(error, "채팅 목록을 불러오지 못했습니다."));
    }
  };

  const loadChatDetail = async (chatId: number) => {
    setIsHistoryLoading(true);

    try {
      const response = await apiFetch(`/api/chat/${chatId}`);
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
          terraformCode: chat.terraform_code || undefined,
          validationSummary: chat.validation_summary || undefined,
          timestamp,
        },
      ]);
    } catch (error) {
      console.error("Failed to load chat detail:", error);
      pushToast(toUserMessage(error, "채팅 내역을 불러오지 못했습니다."));
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
    // Check Health
    apiFetch("/api/health", { timeoutMs: 8000 })
      .then(() => setHealthStatus("ok"))
      .catch(() => setHealthStatus("error"));

    // 소셜 로그인 콜백 실패 시 안내 (?auth_error=...) 후 URL 정리
    const params = new URLSearchParams(window.location.search);
    const authError = params.get("auth_error");
    if (authError) {
      pushToast(
        authError === "email"
          ? "소셜 계정에서 이메일을 가져오지 못했습니다. 이메일 공개 설정을 확인해 주세요."
          : "소셜 로그인에 실패했습니다. 다시 시도해 주세요.",
      );
      window.history.replaceState({}, "", window.location.pathname);
    }

    // Check Auth Token
    fetchUserInfo();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 실제 요구사항 전송 (신규 전송 + 재시도 공용)
  const sendRequirement = async (requirement: string) => {
    const text = requirement.trim();
    if (!text || isLoading) return;

    lastRequirement.current = text;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      text,
      timestamp: new Date(),
    };

    setSelectedChatId(null);
    setMessages([userMessage]);
    setIsLoading(true);

    try {
      const response = await apiFetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ requirements: text }),
      });

      const data = await response.json();
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "bot",
        text: data.message || "생성이 완료되었습니다.",
        imageUrl: data.image_url || undefined,
        terraformCode: data.terraform_code || undefined,
        validationSummary: data.validation_summary || undefined,
        timestamp: new Date(),
      };

      setMessages([userMessage, botMessage]);
      if (typeof data.chat_id === "number") setSelectedChatId(data.chat_id);

      if (currentUser) {
        await loadChatHistory();
      }
    } catch (error) {
      console.error("Error:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "bot",
        text: toUserMessage(error, "생성 중 오류가 발생했습니다."),
        isError: true,
        timestamp: new Date(),
      };
      setMessages([userMessage, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;
    const text = inputValue;
    setInputValue("");
    await sendRequirement(text);
  };

  const handleRetry = () => {
    if (lastRequirement.current) sendRequirement(lastRequirement.current);
  };

  const handleLoginSuccess = (emailStr: string) => {
    setCurrentUser({ id: 0, email: emailStr });
    fetchUserInfo();
  };

  const handleLogout = async () => {
    try {
      await apiFetch("/api/user/logout", { method: "POST" });
    } catch (error) {
      console.error("Failed to logout on backend:", error);
      pushToast(toUserMessage(error, "로그아웃 처리 중 문제가 발생했습니다."));
    }
    setCurrentUser(null);
    setChatHistory([]);
    startNewChat();
  };

  return (
    <main className="flex h-screen bg-slate-50 font-sans text-slate-800">
      {/* Sidebar Component */}
      <Sidebar
        currentUser={currentUser}
        chatHistory={chatHistory}
        selectedChatId={selectedChatId}
        onSelectChat={loadChatDetail}
        onStartNewChat={startNewChat}
        onLogout={handleLogout}
        onOpenAuth={() => setIsAuthModalOpen(true)}
      />

      {/* Main Chat Workspace Component */}
      <ChatArea
        messages={messages}
        isLoading={isLoading}
        isHistoryLoading={isHistoryLoading}
        healthStatus={healthStatus}
        inputValue={inputValue}
        setInputValue={setInputValue}
        onSend={handleSend}
        onRetry={handleRetry}
        onStartNewChat={startNewChat}
        onOpenAuth={() => setIsAuthModalOpen(true)}
        currentUser={currentUser}
      />

      {/* Glassmorphic Auth Modal Component */}
      <AuthModal
        isOpen={isAuthModalOpen}
        onClose={() => setIsAuthModalOpen(false)}
        onLoginSuccess={handleLoginSuccess}
      />

      {/* Toast 알림 */}
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </main>
  );
}
