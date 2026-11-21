import { AlertTriangle, Bot, Clock, Cpu, Database, KeyRound, ShieldAlert, User, Zap } from 'lucide-react';
import type { ChatMessageItem } from '../types';

interface ChatMessageProps {
  message: ChatMessageItem;
}

export function ChatMessage({ message }: ChatMessageProps) {
  if (message.role === 'user') {
    return (
      <div className="msg-row msg-row-user">
        <div className="msg-bubble msg-bubble-user">
          <div className="msg-content">{message.content}</div>
        </div>
        <div className="msg-avatar msg-avatar-user" aria-hidden="true">
          <User size={16} />
        </div>
      </div>
    );
  }

  if (message.role === 'error') {
    return (
      <div className="msg-row msg-row-error" role="alert">
        <div className="msg-avatar msg-avatar-error" aria-hidden="true">
          <AlertTriangle size={16} />
        </div>
        <div className="msg-bubble msg-bubble-error">
          <div className="error-title">
            Request Error{message.statusCode ? ` (HTTP ${message.statusCode})` : ''}
          </div>
          <div className="msg-content">{message.content}</div>
        </div>
      </div>
    );
  }

  const meta = message.metadata;
  const isCache = meta?.resolution_source === 'LLM_CACHE';
  const hadFailover = (meta?.fallback_attempts ?? 0) > 0;

  return (
    <div className="msg-row msg-row-assistant">
      <div className="msg-avatar msg-avatar-assistant" aria-hidden="true">
        <Bot size={16} />
      </div>
      <div className="msg-bubble msg-bubble-assistant" data-testid="assistant-message">
        <div className="msg-content">{message.content}</div>

        {meta && (
          <div className="meta-bar">
            <span className={`meta-pill ${isCache ? 'meta-pill-cache' : 'meta-pill-live'}`}>
              {isCache ? <Database size={12} /> : <Zap size={12} />}
              <span data-testid="meta-source">{meta.resolution_source}</span>
            </span>

            <span className="meta-pill meta-pill-neutral" title="Model used">
              <Cpu size={12} />
              <span data-testid="meta-model">{meta.model}</span>
            </span>

            <span className="meta-pill meta-pill-neutral" title="Server latency">
              <Clock size={12} />
              <span data-testid="meta-latency">{meta.latency_ms} ms</span>
            </span>

            <span className="meta-pill meta-pill-neutral" title="Prompt tokens for this request">
              <span className="meta-label">In:</span>
              <span data-testid="meta-input-tokens">{meta.input_tokens}</span>
            </span>

            <span className="meta-pill meta-pill-neutral" title="Completion tokens for this reply">
              <span className="meta-label">Out:</span>
              <span data-testid="meta-output-tokens">{meta.output_tokens}</span>
            </span>

            {meta.key_id && (
              <span
                className={`meta-pill ${hadFailover ? 'meta-pill-failover' : 'meta-pill-neutral'}`}
                title="API key ID that answered"
              >
                {hadFailover ? <ShieldAlert size={12} /> : <KeyRound size={12} />}
                <span>
                  {meta.key_id}
                  {hadFailover ? ` · ${meta.fallback_attempts} failover` : ''}
                </span>
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
