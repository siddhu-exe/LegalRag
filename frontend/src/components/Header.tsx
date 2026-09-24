import React from 'react';
import { BackendTarget } from '../types';

interface HeaderProps {
  backend: BackendTarget;
  onBackendChange: (target: BackendTarget) => void;
  onNewSearch?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  backend,
  onBackendChange,
  onNewSearch,
}) => {
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

        {/* Right Action Bar: Pipeline Badge & Backend Switcher */}
        <div className="flex items-center space-x-3 sm:space-x-4">
          {/* Architecture Badge */}
          <div className="hidden md:flex items-center space-x-1.5 px-2.5 py-1 rounded bg-canvas-surfaceLow border border-white/[0.08] text-[11px] font-mono text-slateSteel-light">
            <span className="w-1.5 h-1.5 rounded-full bg-vectorMint animate-pulse"></span>
            <span>Hybrid RRF + Reranker</span>
          </div>

          {/* Runtime Backend Switcher */}
          <div className="flex items-center bg-canvas-surfaceLow p-1 rounded-md border border-white/[0.08] text-xs font-mono">
            <button
              type="button"
              onClick={() => onBackendChange('local')}
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded transition-all ${
                backend === 'local'
                  ? 'bg-canvas-surfaceHigh text-brass-light font-medium shadow-sm border border-brass/20'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
              title="Query local FastAPI instance on port 7860"
            >
              <span className={`w-1.5 h-1.5 rounded-full ${backend === 'local' ? 'bg-brass animate-pulse' : 'bg-gray-600'}`}></span>
              <span>Local</span>
              <span className="text-[10px] text-gray-500 hidden sm:inline">:7860</span>
            </button>

            <button
              type="button"
              onClick={() => onBackendChange('deployed')}
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded transition-all ${
                backend === 'deployed'
                  ? 'bg-canvas-surfaceHigh text-brass-light font-medium shadow-sm border border-brass/20'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
              title="Query deployed remote backend endpoint"
            >
              <span className={`w-1.5 h-1.5 rounded-full ${backend === 'deployed' ? 'bg-vectorMint animate-pulse' : 'bg-gray-600'}`}></span>
              <span>Deployed</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
