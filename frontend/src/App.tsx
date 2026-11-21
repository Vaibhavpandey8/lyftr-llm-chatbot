import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react';
import { Loader2, Menu, Send, Sparkles } from 'lucide-react';
import { ChatMessage } from './components/ChatMessage';
import { Sidebar } from './components/Sidebar';
import { ApiError, fetchHealth, sendChatMessage } from './lib/api';
import type { ChatMessageItem, HealthResponse } from './types';

function generateSessionId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return `s-${crypto.randomUUID().slice(0, 8)}`;
  }
  return `s-${Math.random().toString(36).slice(2, 10)}`;
}

const STARTER_PROMPTS = [
  'What is the capital of France?',
  'Explain API key failover in two sentences.',
  'Give me 3 concise tips for SQLite in production.',
];

export default function App() {
  // Always start a fresh session on every page load
  const [sessionId, setSessionId] = useState<string>(() => generateSessionId());
  const [messages, setMessages] = useState<ChatMessageItem[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [simulateFailure, setSimulateFailure] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const bottomRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    let active = true;
    fetchHealth()
      .then((res) => {
        if (active) {
          setHealth(res);
          setHealthError(null);
        }
      })
      .catch((err: unknown) => {
        if (active) {
          setHealth(null);
          setHealthError(err instanceof Error ? err.message : 'Backend unreachable');
        }
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isSending]);

  const handleNewSession = () => {
    setSessionId(generateSessionId());
    setMessages([]);
    setInputValue('');
    inputRef.current?.focus();
  };

  const submitMessage = async (rawText: string) => {
    if (isSending) return;

    const trimmed = rawText.trim();
    if (!trimmed) {
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: 'error',
          content: 'Please enter a non-empty message before sending.',
          statusCode: 422,
          timestamp: new Date().toISOString(),
        },
      ]);
      return;
    }

    // Clear input immediately on send (satisfies chat-input empty-after-sending contract)
    setInputValue('');

    const userItem: ChatMessageItem = {
      id: `user-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      role: 'user',
      content: trimmed,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userItem]);
    setIsSending(true);

    try {
      const response = await sendChatMessage({
        sessionId,
        message: trimmed,
        systemPrompt: systemPrompt.trim() || undefined,
        simulateFailure: simulateFailure || undefined,
      });

      const assistantItem: ChatMessageItem = {
        id: `asst-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        role: 'assistant',
        content: response.reply,
        metadata: response.metadata,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantItem]);
    } catch (err: unknown) {
      const statusCode = err instanceof ApiError ? err.status : undefined;
      const errorText =
        err instanceof Error ? err.message : 'An unexpected error occurred while contacting the server.';

      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
          role: 'error',
          content: errorText,
          statusCode: statusCode && statusCode > 0 ? statusCode : undefined,
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setIsSending(false);
      inputRef.current?.focus();
    }
  };

  const handleFormSubmit = (e: FormEvent) => {
    e.preventDefault();
    void submitMessage(inputValue);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void submitMessage(inputValue);
    }
  };

  return (
    <div className="app-shell">
      <Sidebar
        sessionId={sessionId}
        systemPrompt={systemPrompt}
        onSystemPromptChange={setSystemPrompt}
        simulateFailure={simulateFailure}
        onSimulateFailureChange={setSimulateFailure}
        onNewSession={handleNewSession}
        health={health}
        healthError={healthError}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <main className="chat-main">
        <header className="chat-header">
          <div className="chat-header-left">
            <button
              type="button"
              className="menu-btn"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open settings"
            >
              <Menu size={18} />
            </button>
            <div>
              <h1 className="chat-title">LLM Chatbot Gateway</h1>
              <p className="chat-subtitle">
                Multi-turn SQLite history · LiteLLM response cache · Automatic key failover
              </p>
            </div>
          </div>

          <div className="chat-header-right">
            {simulateFailure && (
              <span className="sim-active-badge" title="Active X-Simulate-Failure header">
                Simulating: {simulateFailure}
              </span>
            )}
            <span className="header-session-pill">{sessionId}</span>
          </div>
        </header>

        <section className="chat-feed" aria-live="polite">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">
                <Sparkles size={26} />
              </div>
              <h2>Ready for Resilient Inference</h2>
              <p>
                Every reply displays its exact resolution source (<code>LIVE_INFERENCE</code> or{' '}
                <code>LLM_CACHE</code>), model name, latency in milliseconds, and per-request token
                counts.
              </p>
              <div className="starter-grid">
                {STARTER_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    className="starter-chip"
                    onClick={() => void submitMessage(prompt)}
                    disabled={isSending}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages-list">
              {messages.map((msg) => (
                <ChatMessage key={msg.id} message={msg} />
              ))}

              {isSending && (
                <div className="msg-row msg-row-assistant">
                  <div className="msg-avatar msg-avatar-assistant" aria-hidden="true">
                    <Loader2 size={16} className="spin" />
                  </div>
                  <div className="msg-bubble msg-bubble-typing">
                    <span>Generating response...</span>
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          )}
        </section>

        <footer className="composer-wrap">
          <form className="composer-form" onSubmit={handleFormSubmit}>
            <textarea
              ref={inputRef}
              data-testid="chat-input"
              className="composer-input"
              rows={1}
              placeholder="Ask anything... (Press Enter to send, Shift+Enter for new line)"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isSending}
              maxLength={4000}
            />
            <button
              type="submit"
              data-testid="send-button"
              className="send-btn"
              disabled={isSending}
              aria-label="Send message"
            >
              {isSending ? <Loader2 size={18} className="spin" /> : <Send size={18} />}
              <span>Send</span>
            </button>
          </form>
        </footer>
      </main>
    </div>
  );
}
