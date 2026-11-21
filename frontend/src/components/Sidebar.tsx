import { Activity, Database, KeyRound, Plus, Settings2, ShieldAlert } from 'lucide-react';
import type { HealthResponse } from '../types';

interface SidebarProps {
  sessionId: string;
  systemPrompt: string;
  onSystemPromptChange: (value: string) => void;
  simulateFailure: string;
  onSimulateFailureChange: (value: string) => void;
  onNewSession: () => void;
  health: HealthResponse | null;
  healthError: string | null;
  isOpen: boolean;
  onClose: () => void;
}

const FAILOVER_PRESETS = [
  { label: 'Off (Normal Routing)', value: '' },
  { label: 'key_1=429 (Rate Limit -> key_2)', value: 'key_1=429' },
  { label: 'key_1=500 (Server Error -> key_2)', value: 'key_1=500' },
  { label: '*=429 (All Keys Fail -> 503)', value: '*=429' },
];

export function Sidebar({
  sessionId,
  systemPrompt,
  onSystemPromptChange,
  simulateFailure,
  onSimulateFailureChange,
  onNewSession,
  health,
  healthError,
  isOpen,
  onClose,
}: SidebarProps) {
  return (
    <>
      {isOpen && <div className="sidebar-backdrop" onClick={onClose} />}
      <aside className={`sidebar ${isOpen ? 'sidebar-open' : ''}`}>
        <div className="sidebar-header">
          <div className="brand">
            <div className="brand-mark">L</div>
            <div>
              <div className="brand-name">Lyftr AI</div>
              <div className="brand-sub">Resilient LLM Gateway</div>
            </div>
          </div>
          <button
            type="button"
            className="btn-new-session"
            onClick={() => {
              onNewSession();
              onClose();
            }}
            title="Start a brand-new session (useful for testing cross-session cache hits)"
          >
            <Plus size={15} />
            New Session
          </button>
        </div>

        <div className="sidebar-section">
          <div className="section-label">
            <Activity size={13} />
            Active Session
          </div>
          <div className="session-id-box" title={sessionId}>
            <code>{sessionId}</code>
          </div>
          <p className="section-hint">
            A new session starts on every page load. Send the same first prompt in a new session to
            observe an instant <code>LLM_CACHE</code> hit.
          </p>
        </div>

        <div className="sidebar-section">
          <label className="section-label" htmlFor="system-prompt-textarea">
            <Settings2 size={13} />
            System Prompt (Optional)
          </label>
          <textarea
            id="system-prompt-textarea"
            data-testid="system-prompt-input"
            className="sidebar-textarea"
            rows={4}
            placeholder="e.g. You are a concise senior Python engineer..."
            value={systemPrompt}
            onChange={(e) => onSystemPromptChange(e.target.value)}
          />
          <p className="section-hint">Overrides the default system prompt for this session.</p>
        </div>

        <div className="sidebar-section">
          <label className="section-label" htmlFor="failover-sim-select">
            <ShieldAlert size={13} />
            Failover Test Hook
          </label>
          <select
            id="failover-sim-select"
            className="sidebar-select"
            value={simulateFailure}
            onChange={(e) => onSimulateFailureChange(e.target.value)}
          >
            {FAILOVER_PRESETS.map((preset) => (
              <option key={preset.label} value={preset.value}>
                {preset.label}
              </option>
            ))}
          </select>
          <p className="section-hint">
            Attaches <code>X-Simulate-Failure</code> header to demonstrate live key failover.
          </p>
        </div>

        <div className="sidebar-footer">
          <div className="health-card">
            <div className="health-row">
              <span className="health-title">Gateway Status</span>
              <span className={`status-dot ${health ? 'status-ok' : 'status-err'}`}>
                {health ? 'Connected' : healthError ? 'Offline' : 'Checking...'}
              </span>
            </div>
            {health ? (
              <div className="health-metrics">
                <span className="health-chip">
                  <KeyRound size={12} />
                  {health.keys_configured} keys
                </span>
                <span className="health-chip">
                  <Database size={12} />
                  Cache {health.cache_enabled ? 'ON' : 'OFF'}
                </span>
              </div>
            ) : (
              healthError && <div className="health-err-text">{healthError}</div>
            )}
          </div>
        </div>
      </aside>
    </>
  );
}
