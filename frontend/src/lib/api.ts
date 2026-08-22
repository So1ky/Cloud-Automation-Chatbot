// 백엔드 호출 공통 헬퍼: 타임아웃(AbortController) + 에러 정규화(네트워크/타임아웃/HTTP).
// 모든 fetch를 이 헬퍼로 통일해 일관된 에러 처리를 제공한다.

export const API_BASE = "http://localhost:8000";

export type ApiErrorKind = "network" | "timeout" | "http";

export class ApiError extends Error {
  kind: ApiErrorKind;
  status: number | null;
  detail: string | null;

  constructor(
    kind: ApiErrorKind,
    message: string,
    status: number | null = null,
    detail: string | null = null,
  ) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.detail = detail;
  }
}

interface ApiFetchOptions extends RequestInit {
  timeoutMs?: number;
}

// FastAPI 에러 detail 파싱 (문자열 또는 [{msg}, ...] 검증 오류 형태 모두 대응)
function parseDetail(data: unknown): string | null {
  if (!data || typeof data !== "object") return null;
  const detail = (data as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const msgs = detail
      .map((d) =>
        d && typeof d === "object" && "msg" in d
          ? String((d as { msg: unknown }).msg)
          : null,
      )
      .filter(Boolean);
    if (msgs.length) return msgs.join(" ");
  }
  return null;
}

// 백엔드 요청. 실패 시 항상 ApiError를 던진다.
export async function apiFetch(
  path: string,
  options: ApiFetchOptions = {},
): Promise<Response> {
  // 파이프라인 응답이 오래 걸릴 수 있어 기본 타임아웃을 넉넉히(3분) 잡는다.
  const { timeoutMs = 180000, ...init } = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      credentials: "include",
      ...init,
      signal: controller.signal,
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new ApiError(
        "timeout",
        "요청 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.",
      );
    }
    throw new ApiError(
      "network",
      "백엔드 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해 주세요.",
    );
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    let detail: string | null = null;
    try {
      detail = parseDetail(await response.clone().json());
    } catch {
      detail = null;
    }
    throw new ApiError(
      "http",
      detail || `요청이 실패했습니다. (오류 ${response.status})`,
      response.status,
      detail,
    );
  }

  return response;
}

// 에러를 상황에 맞는 한국어 사용자 메시지로 변환한다.
export function toUserMessage(
  err: unknown,
  fallback = "요청 처리 중 오류가 발생했습니다.",
): string {
  if (err instanceof ApiError) {
    if (err.kind === "network" || err.kind === "timeout") return err.message;
    switch (err.status) {
      case 400:
        return `요청을 확인해 주세요. ${err.detail ?? ""}`.trim();
      case 401:
        return "로그인이 필요합니다.";
      case 403:
        return "이 작업을 수행할 권한이 없습니다.";
      case 404:
        return "요청한 항목을 찾을 수 없습니다.";
      case 429:
        return "요청이 많거나 사용 한도를 초과했습니다. 잠시 후 다시 시도해 주세요.";
      case 502:
        return "AI 모델 호출에 실패했습니다 (사용 한도·크레딧 또는 일시적 오류). 잠시 후 다시 시도해 주세요.";
      case 504:
        return "처리 시간이 초과되었습니다. 다시 시도해 주세요.";
      case 500:
        return err.detail
          ? `서버 오류가 발생했습니다: ${err.detail}`
          : "서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.";
      default:
        return err.detail || fallback;
    }
  }
  if (err instanceof Error) return err.message || fallback;
  return fallback;
}
