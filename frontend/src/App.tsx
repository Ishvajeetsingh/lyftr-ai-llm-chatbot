import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import { fetchHealth, sendChatMessage } from './api';
import { MessageItem, HealthResponse } from './types';

function generateSessionId(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return `sess-${crypto.randomUUID()}`;
  }
  return `sess-${Math.random().toString(36).substring(2, 12)}-${Date.now()}`;
}

export const App: React.FC = () => {
  const [sessionId, setSessionId] = useState<string>(() => generateSessionId());
  const [systemPrompt, setSystemPrompt] = useState<string>('');
  const [inputText, setInputText] = useState<string>('');
  const [messages, setMessages] = useState<MessageItem[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);
  const [failureSimulation, setFailureSimulation] = useState<string>('none');
  const [health, setHealth] = useState<HealthResponse | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  useEffect(() => {
    fetchHealth()
      .then((data) => setHealth(data))
      .catch((err) => console.warn('Health check error:', err));
  }, []);

  const handleNewSession = () => {
    setSessionId(generateSessionId());
    setMessages([]);
    setErrorBanner(null);
    setInputText('');
  };

  const getSimHeaderValue = (): string | undefined => {
    if (failureSimulation === 'none') return undefined;
    return failureSimulation;
  };

  const handleSendMessage = async () => {
    const trimmed = inputText.trim();
    if (!trimmed || loading) return;

    // Clear input immediately after send as required
    setInputText('');
    setErrorBanner(null);

    const userMsg: MessageItem = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: trimmed,
      created_at: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const response = await sendChatMessage(
        sessionId,
        trimmed,
        systemPrompt || undefined,
        getSimHeaderValue()
      );

      const assistantMsg: MessageItem = {
        id: `a-${Date.now()}`,
        role: 'assistant',
        content: response.reply,
        metadata: response.metadata,
        created_at: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      const errorText = err.message || 'Failed to complete request';
      setErrorBanner(`Error: ${errorText}`);

      const errorMsgItem: MessageItem = {
        id: `e-${Date.now()}`,
        role: 'error',
        content: `⚠️ Error (${err.status || '503/422'}): ${errorText}`,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMsgItem]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar Controls */}
      <aside className="sidebar">
        <div className="brand-header">
          <div className="logo-badge">⚡ Lyftr AI</div>
          <h1>LLM Chatbot</h1>
        </div>

        <button className="btn-new-session" onClick={handleNewSession}>
          ➕ New Session
        </button>

        <div className="sidebar-section">
          <label>Active Session ID</label>
          <div className="session-info">{sessionId}</div>
        </div>

        <div className="sidebar-section">
          <label>Session System Prompt</label>
          <textarea
            data-testid="system-prompt-input"
            className="system-prompt-textarea"
            placeholder="Optional system prompt..."
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
          />
        </div>

        <div className="sidebar-section">
          <label>Failure Simulation (Header)</label>
          <select
            className="sim-select"
            value={failureSimulation}
            onChange={(e) => setFailureSimulation(e.target.value)}
          >
            <option value="none">None (Standard Inference)</option>
            <option value="key_1=429">Simulate key_1=429 (Rate Limit)</option>
            <option value="key_1=500">Simulate key_1=500 (Server Error)</option>
            <option value="key_1=429, key_2=500">Simulate key_1=429, key_2=500</option>
            <option value="*=429">Simulate *=429 (All Keys Fail 503)</option>
          </select>
        </div>

        <div className="status-card">
          <div className="status-row">
            <span>Backend Status:</span>
            <span className="status-badge">{health?.status || 'connecting...'}</span>
          </div>
          <div className="status-row">
            <span>Keys Configured:</span>
            <span>{health ? health.keys_configured : '-'}</span>
          </div>
          <div className="status-row">
            <span>LiteLLM Cache:</span>
            <span>{health?.cache_enabled ? 'Active' : '-'}</span>
          </div>
        </div>
      </aside>

      {/* Main Chat Interface */}
      <main className="chat-main">
        <header className="chat-header">
          <div className="chat-header-title">Multi-Turn Session with Key Failover & Caching</div>
          <div className="chat-header-meta">Session: {sessionId.substring(0, 18)}...</div>
        </header>

        {errorBanner && (
          <div className="error-banner">
            <span>{errorBanner}</span>
            <button
              onClick={() => setErrorBanner(null)}
              style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer' }}
            >
              ✕
            </button>
          </div>
        )}

        <div className="messages-area">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">💬</div>
              <p>Type a message below to start your conversation.</p>
              <small>Identical first prompts in new sessions will hit LiteLLM Cache!</small>
            </div>
          ) : (
            messages.map((msg) => {
              if (msg.role === 'user') {
                return (
                  <div key={msg.id} className="message-row user">
                    <div className="message-bubble user">
                      <div className="message-text">{msg.content}</div>
                    </div>
                  </div>
                );
              }

              if (msg.role === 'error') {
                return (
                  <div key={msg.id} className="message-row error">
                    <div className="message-bubble error">
                      <div className="message-text">{msg.content}</div>
                    </div>
                  </div>
                );
              }

              // Assistant Message
              return (
                <div key={msg.id} className="message-row assistant">
                  <div data-testid="assistant-message" className="message-bubble assistant">
                    <div className="message-text">{msg.content}</div>

                    {msg.metadata && (
                      <div className="metadata-bar">
                        <span
                          data-testid="meta-source"
                          className={`meta-badge ${msg.metadata.resolution_source.toLowerCase()}`}
                        >
                          {msg.metadata.resolution_source}
                        </span>

                        <span data-testid="meta-model" className="meta-badge model">
                          {msg.metadata.model}
                        </span>

                        <span data-testid="meta-latency" className="meta-badge latency">
                          {msg.metadata.latency_ms} ms
                        </span>

                        <span className="token-group">
                          In: <span data-testid="meta-input-tokens">{msg.metadata.input_tokens}</span>
                        </span>

                        <span className="token-group">
                          Out: <span data-testid="meta-output-tokens">{msg.metadata.output_tokens}</span>
                        </span>

                        {msg.metadata.key_id && (
                          <span className="meta-badge key-id">
                            Key: {msg.metadata.key_id}
                          </span>
                        )}

                        {msg.metadata.fallback_attempts > 0 && (
                          <span className="meta-badge fallback">
                            Fallback attempts: {msg.metadata.fallback_attempts}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}

          {loading && (
            <div className="message-row assistant">
              <div className="message-bubble assistant" style={{ opacity: 0.7 }}>
                <div className="message-text">Thinking & verifying key failover...</div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Chat Input Bar */}
        <div className="input-bar">
          <textarea
            data-testid="chat-input"
            className="chat-input"
            rows={1}
            placeholder="Type your message (Enter to send, Shift+Enter for new line)..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <button
            data-testid="send-button"
            className="send-button"
            onClick={handleSendMessage}
            disabled={loading || !inputText.trim()}
          >
            Send
          </button>
        </div>
      </main>
    </div>
  );
};

export default App;
