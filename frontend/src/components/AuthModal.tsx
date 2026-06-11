"use client";
import React, { useState } from "react";

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLoginSuccess: (email: string) => void;
}

export default function AuthModal({
  isOpen,
  onClose,
  onLoginSuccess,
}: AuthModalProps) {
  const [authTab, setAuthTab] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [passwordConfirm, setPasswordConfirm] = useState("");
  const [authError, setAuthError] = useState("");
  const [authSuccess, setAuthSuccess] = useState("");
  const [isAuthLoading, setIsAuthLoading] = useState(false);

  if (!isOpen) return null;

  const formatErrorMessage = (detail: any): string => {
    if (typeof detail === "string") {
      return detail;
    }
    if (Array.isArray(detail)) {
      return detail
        .map((err: any) => {
          if (err.loc && err.loc.includes("email")) {
            return "올바른 이메일 주소 형식을 입력해 주세요 (예: test@test.com).";
          }
          return err.msg || "입력값이 올바르지 않습니다.";
        })
        .join("\n");
    }
    return "요청을 처리할 수 없습니다.";
  };

  // HTTP 응답 검증 및 에러 메시지 추출 헬퍼
  const validateResponse = async (response: Response, defaultMessage: string) => {
    if (response.ok) return;

    let errorMessage = defaultMessage;
    try {
      const errorData = await response.json();
      errorMessage = formatErrorMessage(errorData.detail) || errorMessage;
    } catch {
      errorMessage = response.status === 500
        ? "서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
        : `서버 오류가 발생했습니다. (오류 코드: ${response.status})`;
    }
    throw new Error(errorMessage);
  };

  // 예외 객체를 사용자 친화적 메시지로 변환하는 헬퍼
  const getFriendlyErrorMessage = (error: any, fallbackMessage: string): string => {
    if (error instanceof Error && error.message === "Failed to fetch") {
      return "서버와 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인해 주세요.";
    }
    return error.message || fallbackMessage;
  };

  const handleLogin = async (e: React.SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault();
    setAuthError("");
    setAuthSuccess("");
    setIsAuthLoading(true);
    try {
      const response = await fetch("http://localhost:8000/api/user/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
        credentials: "include",
      });
      
      await validateResponse(response, "로그인에 실패했습니다.");
      
      onLoginSuccess(email);
      setEmail("");
      setPassword("");
      onClose();
    } catch (error: any) {
      setAuthError(getFriendlyErrorMessage(error, "로그인 중 오류가 발생했습니다."));
    } finally {
      setIsAuthLoading(false);
    }
  };

  const handleSignup = async (e: React.SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault();
    setAuthError("");
    setAuthSuccess("");
    if (password !== passwordConfirm) {
      setAuthError("비밀번호가 일치하지 않습니다.");
      return;
    }
    setIsAuthLoading(true);
    try {
      const response = await fetch("http://localhost:8000/api/user/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email,
          password1: password,
          password2: passwordConfirm,
        }),
        credentials: "include",
      });

      await validateResponse(response, "회원가입에 실패했습니다.");

      const data = await response.json();
      setAuthSuccess(data.message || "회원가입이 성공적으로 완료되었습니다.");
      setEmail("");
      setPassword("");
      setPasswordConfirm("");
      
      // 2초 후 로그인 탭으로 전환 및 메시지 초기화
      setTimeout(() => {
        setAuthTab("login");
        setAuthSuccess("");
      }, 2000);
    } catch (error: any) {
      setAuthError(getFriendlyErrorMessage(error, "회원가입 중 오류가 발생했습니다."));
    } finally {
      setIsAuthLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm transition-opacity">
      <div
        className="w-full max-w-md overflow-hidden rounded-2xl bg-white shadow-2xl transition-all border border-slate-100"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="relative border-b border-slate-100 p-6 flex flex-col items-center">
          <button
            onClick={() => {
              onClose();
              setAuthError("");
              setAuthSuccess("");
            }}
            className="absolute top-4 right-4 rounded-full p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700 transition-colors"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>

          <h2 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            Cloud Diagram Bot
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            대화 기록을 동기화하고 간직하세요
          </p>

          {/* Tab Selector */}
          <div className="flex w-full mt-6 rounded-lg bg-slate-100 p-1">
            <button
              type="button"
              onClick={() => {
                setAuthTab("login");
                setAuthError("");
                setAuthSuccess("");
              }}
              className={`flex-1 rounded-md py-2 text-center text-xs font-bold transition-all ${
                authTab === "login"
                  ? "bg-white text-blue-600 shadow-sm"
                  : "text-slate-500 hover:text-slate-800"
              }`}
            >
              로그인
            </button>
            <button
              type="button"
              onClick={() => {
                setAuthTab("signup");
                setAuthError("");
                setAuthSuccess("");
              }}
              className={`flex-1 rounded-md py-2 text-center text-xs font-bold transition-all ${
                authTab === "signup"
                  ? "bg-white text-blue-600 shadow-sm"
                  : "text-slate-500 hover:text-slate-800"
              }`}
            >
              회원가입
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <form
          onSubmit={authTab === "login" ? handleLogin : handleSignup}
          className="p-6 space-y-4"
        >
          {authError && (
            <div className="rounded-lg bg-red-50 p-3 text-xs font-medium text-red-600 flex items-center gap-2 border border-red-100">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-4 w-4 shrink-0"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
              <span className="whitespace-pre-line">{authError}</span>
            </div>
          )}

          {authSuccess && (
            <div className="rounded-lg bg-green-50 p-3 text-xs font-medium text-green-600 flex items-center gap-2 border border-green-100 animate-fadeIn">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-4 w-4 shrink-0"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span>{authSuccess}</span>
            </div>
          )}

          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-500">
              이메일 주소
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="name@company.com"
              className="w-full rounded-xl border border-slate-200 px-4 py-2.5 text-sm outline-none transition-all focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
              required
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-500">
              비밀번호
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className="w-full rounded-xl border border-slate-200 px-4 py-2.5 text-sm outline-none transition-all focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
              required
            />
          </div>

          {authTab === "signup" && (
            <div className="space-y-1.5 animate-fadeIn">
              <label className="text-xs font-semibold text-slate-500">
                비밀번호 확인
              </label>
              <input
                type="password"
                value={passwordConfirm}
                onChange={(e) => setPasswordConfirm(e.target.value)}
                placeholder="••••••••"
                className="w-full rounded-xl border border-slate-200 px-4 py-2.5 text-sm outline-none transition-all focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
                required
              />
            </div>
          )}

          <button
            type="submit"
            disabled={isAuthLoading}
            className="w-full rounded-xl bg-blue-600 py-3 text-sm font-bold text-white shadow-md transition-all hover:bg-blue-700 active:scale-[0.98] disabled:bg-slate-300"
          >
            {isAuthLoading ? (
              <div className="flex items-center justify-center gap-2">
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                <span>요청 처리 중...</span>
              </div>
            ) : authTab === "login" ? (
              "로그인하기"
            ) : (
              "회원가입하기"
            )}
          </button>

          <div className="text-center mt-3">
            <button
              type="button"
              onClick={() => {
                onClose();
                setAuthError("");
                setAuthSuccess("");
              }}
              className="text-xs text-slate-400 hover:text-slate-600 transition-colors font-medium"
            >
              로그인하지 않고 게스트로 계속하기
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
