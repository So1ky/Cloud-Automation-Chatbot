"use client";
import React from "react";

interface ChatHistoryItem {
  id: number;
  requirements: string;
  created_at: string;
}

interface SidebarProps {
  currentUser: { id: number; email: string } | null;
  chatHistory: ChatHistoryItem[];
  selectedChatId: number | null;
  onSelectChat: (id: number) => void;
  onStartNewChat: () => void;
  onLogout: () => void;
  onOpenAuth: () => void;
}

export default function Sidebar({
  currentUser,
  chatHistory,
  selectedChatId,
  onSelectChat,
  onStartNewChat,
  onLogout,
  onOpenAuth,
}: SidebarProps) {
  return (
    <aside className="hidden w-80 shrink-0 border-r border-slate-200 bg-white md:flex md:flex-col shadow-sm">
      {/* Sidebar Header */}
      <div className="border-b border-slate-100 p-5 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <span className="text-lg font-bold tracking-tight bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            Cloud Diagram Bot
          </span>
          <button
            onClick={onStartNewChat}
            className="rounded-full bg-blue-50 p-2 text-blue-600 transition-all hover:bg-blue-100 hover:scale-105"
            title="새 채팅 시작"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z" clipRule="evenodd" />
            </svg>
          </button>
        </div>
        <button
          onClick={onStartNewChat}
          className="flex items-center justify-center gap-2 w-full rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow-md transition-all hover:bg-blue-700 hover:shadow-lg active:scale-[0.98]"
        >
          새 대화 시작하기
        </button>
      </div>

      {/* Sidebar Middle (Chat History) */}
      <div className="flex-1 overflow-y-auto p-4">
        {currentUser ? (
          chatHistory.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-48 text-center text-slate-400">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 mb-2 opacity-60" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
              <p className="text-sm">저장된 대화 내역이 없습니다.</p>
            </div>
          ) : (
            <div className="space-y-1">
              <h3 className="px-2 text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">이전 디자인 기록</h3>
              {chatHistory.map((chat) => (
                <button
                  key={chat.id}
                  onClick={() => onSelectChat(chat.id)}
                  className={`w-full rounded-xl px-4 py-3 text-left transition-all ${
                    selectedChatId === chat.id
                      ? "bg-blue-50 text-blue-700 border-l-4 border-blue-600 shadow-sm"
                      : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                  }`}
                >
                  <span className="block truncate text-sm font-medium">
                    {chat.requirements}
                  </span>
                  <span className="mt-1 block text-[10px] text-slate-400">
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
          )
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-center px-4 py-6">
            <div className="rounded-full bg-slate-100 p-4 mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
            </div>
            <h4 className="text-sm font-bold text-slate-700 mb-1">히스토리 저장 비활성화</h4>
            <p className="text-xs text-slate-400 leading-relaxed mb-5">
              로그인 하시면 이전에 생성했던 클라우드 다이어그램 히스토리를 언제든 다시 확인할 수 있습니다.
            </p>
            <button
              onClick={onOpenAuth}
              className="w-full rounded-xl border border-slate-200 bg-white px-4 py-2 text-xs font-semibold text-slate-700 shadow-sm hover:bg-slate-50 active:scale-[0.98]"
            >
              로그인 / 회원가입
            </button>
          </div>
        )}
      </div>

      {/* Sidebar Footer (User Info Section) */}
      <div className="border-t border-slate-100 p-4 bg-slate-50/50">
        {currentUser ? (
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="h-9 w-9 rounded-full bg-blue-100 flex items-center justify-center font-bold text-blue-700 text-sm">
                {currentUser.email.substring(0, 2).toUpperCase()}
              </div>
              <div className="min-w-0">
                <span className="block text-xs text-slate-400 font-medium">Logged in as</span>
                <span className="block text-sm font-semibold text-slate-700 truncate">{currentUser.email}</span>
              </div>
            </div>
            <button
              onClick={onLogout}
              className="rounded-lg p-2 text-slate-400 hover:text-red-500 hover:bg-red-50 transition-colors"
              title="로그아웃"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-full bg-slate-200 flex items-center justify-center font-bold text-slate-500 text-sm">
              G
            </div>
            <div className="flex-1">
              <span className="block text-sm font-bold text-slate-700">게스트 모드</span>
              <span className="block text-[10px] text-slate-400">일회성 세션 활성화됨</span>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
