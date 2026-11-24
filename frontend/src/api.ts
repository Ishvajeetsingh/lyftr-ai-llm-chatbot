import { ChatResponse, HealthResponse, MessageItem } from './types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed with status ${res.status}`);
  }
  return res.json();
}

export async function sendChatMessage(
  sessionId: string,
  message: string,
  systemPrompt?: string,
  simulateFailureHeader?: string
): Promise<ChatResponse> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  if (simulateFailureHeader && simulateFailureHeader.trim() !== '') {
    headers['X-Simulate-Failure'] = simulateFailureHeader.trim();
  }

  const payload: { session_id: string; message: string; system_prompt?: string } = {
    session_id: sessionId,
    message,
  };
  if (systemPrompt && systemPrompt.trim() !== '') {
    payload.system_prompt = systemPrompt.trim();
  }

  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
  });

  const data = await res.json().catch(() => null);

  if (!res.ok) {
    let errorMsg = `Error ${res.status}`;
    if (data && data.detail) {
      if (typeof data.detail === 'string') {
        errorMsg = data.detail;
      } else {
        errorMsg = JSON.stringify(data.detail);
      }
    }
    const err = new Error(errorMsg);
    (err as any).status = res.status;
    (err as any).data = data;
    throw err;
  }

  return data as ChatResponse;
}

export async function fetchSessionMessages(sessionId: string): Promise<MessageItem[]> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/messages`);
  if (res.status === 404) {
    return [];
  }
  if (!res.ok) {
    throw new Error(`Failed to fetch session messages: status ${res.status}`);
  }
  const data = await res.json();
  return data.messages || [];
}
