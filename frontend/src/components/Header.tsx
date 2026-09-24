import React from 'react';
import { BackendStatus } from '../types';

interface HeaderProps {
  onNewSearch?: () => void;
  backendStatus?: BackendStatus;
}

export const Header: React.FC<HeaderProps> = ({ onNewSearch, backendStatus = 'ready' }) => {
  const statusLabel =
    backendStatus === 'ready'
      ? 'Backend ready: hybrid retrieval and reranker online'
      : backendStatus === 'offline'
      ? 'Backend offline: AWS EC2 container in standby'
      : 'Checking backend cluster readiness';

  return (
    <header className="border-b border-white/[0.08] bg-canvas-surface/80 backdrop-blur-xl sticky top-0 z-50 pt-[env(safe-area-inset-top,0px)]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand Logo & Corpus Identification */}
        <button
          type="button"
          onClick={onNewSearch}
          className="flex items-center space-x-3 group text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-brass rounded p-1 min-w-0"
          aria-label="LegalRAG Home — New Judicial Inquiry"
        >
          <div className="w-9 h-9 shrink-0 rounded bg-brass/10 border border-brass/30 flex items-center justify-center text-brass font-serif font-bold text-xl group-hover:bg-brass/20 group-hover:border-brass/50 transition-all shadow-[0_0_15px_-3px_rgba(212,175,55,0.15)]">
            §
          </div>
          <div className="min-w-0">
            <div className="flex items-center space-x-2">
              <span className="font-serif text-lg font-medium tracking-tight text-white group-hover:text-brass-light transition-colors">
                LegalRAG
              </span>
              <span className="hidden sm:inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-canvas-surfaceHigh text-slateSteel-light border border-white/5">
                v1.0
              </span>
            </div>
            <p className="text-[11px] font-mono tracking-wider text-slateSteel uppercase flex items-center space-x-1 truncate">
              <span>100k Judgments</span>
              <span>·</span>
              <span>538k Chunks</span>
            </p>
          </div>
        </button>

        {/* Right Action Bar: Pipeline Architecture & Status Badges */}
        <div className="flex items-center space-x-3 shrink-0">
          {/* Compact status dot for narrow viewports (full pills need ~190px) */}
          <span
            className={`sm:hidden w-2.5 h-2.5 rounded-full ${
              backendStatus === 'ready'
                ? 'bg-vectorMint shadow-[0_0_8px_#2DD4BF]'
                : backendStatus === 'offline'
                ? 'bg-amber-400'
                : 'bg-gray-400 animate-pulse'
            }`}
            role="status"
            aria-label={statusLabel}
          />

          {backendStatus === 'ready' && (
            <div className="hidden sm:inline-flex items-center space-x-2 px-3 py-1.5 rounded-md bg-canvas-surfaceLow border border-white/[0.08] text-xs font-mono text-slateSteel-light">
              <span className="w-2 h-2 rounded-full bg-vectorMint animate-pulse"></span>
              <span>Hybrid RRF + Reranker</span>
            </div>
          )}

          {backendStatus === 'offline' && (
            <div className="hidden sm:inline-flex items-center space-x-2 px-3 py-1.5 rounded-md bg-amber-950/40 border border-amber-500/30 text-xs font-mono text-amber-300">
              <span className="w-2 h-2 rounded-full bg-amber-400"></span>
              <span>AWS EC2 Standby</span>
            </div>
          )}

          {backendStatus === 'checking' && (
            <div className="hidden sm:inline-flex items-center space-x-2 px-3 py-1.5 rounded-md bg-canvas-surfaceLow border border-white/[0.08] text-xs font-mono text-gray-400">
              <span className="w-2 h-2 rounded-full bg-gray-500 animate-pulse"></span>
              <span>Checking Cluster...</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
