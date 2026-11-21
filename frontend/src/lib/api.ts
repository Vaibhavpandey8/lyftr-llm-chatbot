import type { ChatResponse, HealthResponse } from '../types';

const RAW_API_URL = (import.meta.env.VITE_API_URL as string | undefined) || 'http://localhost:8000';
export const API_BASE_URL = RAW_API_URL.replace(/\/+$/, '');

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

function formatErrorDetail(detail: unknown, fallback: string): string {
  if (typeof detail === 'string' && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0] as { msg?: string; loc?: unknown[] };
    if (first && typeof first.msg === 'string') {
      const loc = Array.isArray(first.loc) ? first.loc.slice(1).join('.') : '';
      return loc ? `${loc}: ${first.msg}` : first.msg;
    }
  }
  return fallback;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/health`, {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });

  if (!res.ok) {
    throw new ApiError(`Health check failed (${res.status})`, res.status);
  }
  return (await res.json()) as HealthResponse;
}

export interface SendChatParams {
  sessionId: string;
  message: string;
  systemPrompt?: string;
  simulateFailure?: string;
}

export async function sendChatMessage(params: SendChatParams): Promise<ChatResponse> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  };

  if (params.simulateFailure && params.simulateFailure.trim()) {
    headers['X-Simulate-Failure'] = params.simulateFailure.trim();
  }

  const body: Record<string, string> = {
    session_id: params.sessionId,
    message: params.message,
  };

  if (params.systemPrompt && params.systemPrompt.trim()) {
    body.system_prompt = params.systemPrompt.trim();
  }

  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(
      'Unable to reach the backend server. Please verify the backend is running and CORS is configured.',
      0,
    );
  }

  let data: unknown = null;
  try {
    data = await res.json();
  } catch {
    data = null;
  }

  if (!res.ok) {
    const detail =
      data && typeof data === 'object' && 'detail' in data
        ? (data as { detail: unknown }).detail
        : null;
    const message = formatErrorDetail(
      detail,
      res.status === 503
        ? 'all providers unavailable'
        : `Request failed with status ${res.status}`,
    );
    throw new ApiError(message, res.status);
  }

  return data as ChatResponse;
}
