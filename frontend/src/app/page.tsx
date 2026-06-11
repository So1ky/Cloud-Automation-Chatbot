"use client";
import { useEffect, useState } from "react";
import Sidebar from "@/components/Sidebar";
import ChatArea from "@/components/ChatArea";
import AuthModal from "@/components/AuthModal";

interface Message {
  id: string;
  role: "user" | "bot";
  text: string;
  imageUrl?: string;
  terraformCode?: Record<string, string>;
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

  const fetchUserInfo = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/user/me", {
        credentials: "include",
      });
      if (response.ok) {
        const user = await response.json();
        setCurrentUser(user);
        loadChatHistory();
      } else {
        setCurrentUser(null);
        setChatHistory([]);
      }
    } catch (error) {
      console.error("Failed to fetch user info:", error);
      setCurrentUser(null);
      setChatHistory([]);
    }
  };

  const loadChatHistory = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/chat/history", {
        credentials: "include",
      });
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
      const response = await fetch(`http://localhost:8000/api/chat/${chatId}`, {
        credentials: "include",
      });
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
          terraformCode: chat.terraform_code || undefined,
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
    // Check Health
    fetch("http://localhost:8000/api/health")
      .then((res) => res.json())
      .then(() => setHealthStatus("ok"))
      .catch(() => setHealthStatus("error"));

    // Check Auth Token
    fetchUserInfo();
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
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ requirements: userMessage.text }),
        credentials: "include",
      });

      if (!response.ok) throw new Error("서버 응답 오류");

      const data = await response.json();
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "bot",
        text: data.message || "생성이 완료되었습니다.",
        imageUrl: data.image_url,
        terraformCode: data.terraform_code || undefined,
        timestamp: new Date(),
      };

      setMessages([userMessage, botMessage]);
      setSelectedChatId(data.chat_id);

      if (currentUser) {
        await loadChatHistory();
      }
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

  const handleLoginSuccess = (emailStr: string) => {
    setCurrentUser({ id: 0, email: emailStr });
    fetchUserInfo();
  };

  const handleLogout = async () => {
    try {
      await fetch("http://localhost:8000/api/user/logout", {
        method: "POST",
        credentials: "include",
      });
    } catch (error) {
      console.error("Failed to logout on backend:", error);
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
    </main>
  );
}
