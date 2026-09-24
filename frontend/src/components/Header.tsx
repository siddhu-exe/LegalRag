import React from 'react';

interface HeaderProps {
  onNewSearch?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onNewSearch }) => {
  return (
    <header className="border-b border-white/[0.08] bg-canvas-surface/80 backdrop-blur-xl sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand Logo & Corpus Identification */}
        <button
          type="button"
          onClick={onNewSearch}
          className="flex items-center space-x-3 group text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-brass rounded p-1"
          aria-label="LegalRAG Home — New Judicial Inquiry"
        >
          <div className="w-9 h-9 rounded bg-brass/10 border border-brass/30 flex items-center justify-center text-brass font-serif font-bold text-xl group-hover:bg-brass/20 group-hover:border-brass/50 transition-all shadow-[0_0_15px_-3px_rgba(212,175,55,0.15)]">
            §
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-serif text-lg font-medium tracking-tight text-white group-hover:text-brass-light transition-colors">
                LegalRAG
              </span>
              <span className="hidden sm:inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-canvas-surfaceHigh text-slateSteel-light border border-white/5">
                v1.0
              </span>
            </div>
            <p className="text-[11px] font-mono tracking-wider text-slateSteel uppercase flex items-center space-x-1">
              <span>100k Judgments</span>
              <span>·</span>
              <span>538k Chunks</span>
            </p>
          </div>
        </button>

        {/* Right Action Bar: Pipeline Architecture & Status Badges */}
        <div className="flex items-center space-x-3">
          <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-md bg-canvas-surfaceLow border border-white/[0.08] text-xs font-mono text-slateSteel-light">
            <span className="w-2 h-2 rounded-full bg-vectorMint animate-pulse"></span>
            <span>Hybrid RRF + Reranker</span>
          </div>
        </div>
      </div>
    </header>
  );
};
