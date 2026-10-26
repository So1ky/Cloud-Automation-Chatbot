"use client";
import { useEffect, useRef, useState } from "react";
import Sidebar from "@/components/Sidebar";
import ChatArea from "@/components/ChatArea";
import AuthModal from "@/components/AuthModal";
import ToastContainer, { ToastItem, ToastType } from "@/components/Toast";
import { apiFetch, toUserMessage, ApiError } from "@/lib/api";

export interface CostEstimate {
  skipped?: boolean;
  reason?: string | null;
  total_monthly_cost?: string | null;
  currency?: string;
  potential_yearly_savings?: string | null;
  resources?: { name: string; monthly_cost: string }[];
  finops_issues?: { policy: string; message?: string; resources?: string[] }[];
}

interface Message {
  id: string;
  role: "user" | "bot";
  text: string;
  imageUrl?: string;
  terraformCode?: Record<string, string>;
  validationSummary?: string;
  costEstimate?: CostEstimate;
  isError?: boolean;
  timestamp: Date;
}

interface ChatHistoryItem {
  id: number;
  conversation_id: number;
  requirements: string;
  created_at: string;
}

// GET /api/chat/conversation/{id} 응답의 개별 턴
interface ConversationTurn extends ChatHistoryItem {
  response_message: string;
  image_url?: string | null;
  terraform_code?: Record<string, string> | null;
  validation_summary?: string | null;
  cost_estimate?: CostEstimate | null;
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
  // 현재 진행 중(= 사이드바에서 선택된) 대화 ID. null이면 새 대화
  const [selectedConversationId, setSelectedConversationId] = useState<
    number | null
  >(null);
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

  const loadConversation = async (conversationId: number) => {
    setIsHistoryLoading(true);

    try {
      const response = await apiFetch(`/api/chat/conversation/${conversationId}`);
      const turns: ConversationTurn[] = await response.json();

      setSelectedConversationId(conversationId);
      // 턴 배열(시간순) → user/bot 메시지 쌍으로 전개해 전체 대화를 복원
      setMessages(
        turns.flatMap((turn) => {
          const timestamp = new Date(turn.created_at);
          return [
            {
              id: `${turn.id}-user`,
              role: "user" as const,
              text: turn.requirements,
              timestamp,
            },
            {
              id: `${turn.id}-bot`,
              role: "bot" as const,
              text: turn.response_message,
              imageUrl: turn.image_url || undefined,
              terraformCode: turn.terraform_code || undefined,
              validationSummary: turn.validation_summary || undefined,
              costEstimate: turn.cost_estimate || undefined,
              timestamp,
            },
          ];
        }),
      );
    } catch (error) {
      console.error("Failed to load conversation:", error);
      pushToast(toUserMessage(error, "채팅 내역을 불러오지 못했습니다."));
    } finally {
      setIsHistoryLoading(false);
    }
  };

  const startNewChat = () => {
    setSelectedConversationId(null);
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

    // Check Auth Token (setState는 네트워크 응답 후 실행되므로 동기 재렌더 없음)
    // eslint-disable-next-line react-hooks/set-state-in-effect
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

    // 멀티턴: 대화를 리셋하지 않고 이어붙인다
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await apiFetch("/api/chat/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          requirements: text,
          conversation_id: selectedConversationId,
        }),
      });

      const data = await response.json();
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "bot",
        text: data.message || "생성이 완료되었습니다.",
        imageUrl: data.image_url || undefined,
        terraformCode: data.terraform_code || undefined,
        validationSummary: data.validation_summary || undefined,
        costEstimate: data.cost_estimate || undefined,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, botMessage]);
      if (typeof data.conversation_id === "number")
        setSelectedConversationId(data.conversation_id);

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
      // 실패해도 대화 유지 — conversationId가 남아 있어 재시도 시 같은 대화로 재전송
      setMessages((prev) => [...prev, errorMessage]);
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
        selectedConversationId={selectedConversationId}
        onSelectChat={loadConversation}
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
