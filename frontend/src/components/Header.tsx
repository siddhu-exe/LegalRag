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
    <header className="border-b border-white/10 bg-canvas-surface/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-3 cursor-pointer" onClick={onNewSearch}>
          <div className="w-8 h-8 rounded bg-brass/10 border border-brass/30 flex items-center justify-center text-brass font-serif font-bold text-lg">
            §
          </div>
          <div>
            <h1 className="font-serif text-lg font-semibold tracking-wide text-white">
              LegalRAG
            </h1>
            <p className="text-[10px] font-mono uppercase tracking-widest text-slateSteel-light">
              100k High Court Corpus
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          {/* Backend target toggle pill */}
          <div className="flex items-center bg-canvas-base p-1 rounded border border-white/10 text-xs font-mono">
            <button
              type="button"
              onClick={() => onBackendChange('local')}
              className={`px-2.5 py-1 rounded transition-colors ${
                backend === 'local'
                  ? 'bg-canvas-surfaceHigh text-brass-light font-medium'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              Local
            </button>
            <button
              type="button"
              onClick={() => onBackendChange('deployed')}
              className={`px-2.5 py-1 rounded transition-colors ${
                backend === 'deployed'
                  ? 'bg-canvas-surfaceHigh text-brass-light font-medium'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              Deployed
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
