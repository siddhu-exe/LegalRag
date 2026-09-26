import React, { useState } from 'react';
import {
  RefreshCw,
  ExternalLink,
  ShieldAlert,
  Zap,
} from 'lucide-react';
import { ModelArchitectureInfo } from './ModelArchitectureInfo';

interface BackendOfflineViewProps {
  onRetry: () => Promise<void>;
  isRetrying?: boolean;
}

export const BackendOfflineView: React.FC<BackendOfflineViewProps> = ({
  onRetry,
  isRetrying = false,
}) => {
  const [retryMessage, setRetryMessage] = useState<string | null>(null);

  const handleManualRetry = async () => {
    setRetryMessage(null);
    try {
      await onRetry();
    } catch {
      setRetryMessage('Backend is still offline or starting up.');
      setTimeout(() => setRetryMessage(null), 4000);
    }
  };

  const linkedinUrl = 'https://www.linkedin.com/in/siddharth-dongardive';

  return (
    <div className="relative overflow-hidden min-h-[calc(100dvh-8rem)] py-8 px-4 sm:px-6 lg:px-8">
      {/* Ambient background glow */}
      <div
        className="absolute top-10 left-1/2 -translate-x-1/2 w-[350px] sm:w-[700px] h-[350px] bg-brass/5 rounded-full blur-[160px] pointer-events-none z-0"
        aria-hidden="true"
      />
      <div
        className="absolute top-[600px] right-1/4 w-[300px] sm:w-[500px] h-[300px] bg-vectorMint/5 rounded-full blur-[140px] pointer-events-none z-0"
        aria-hidden="true"
      />

      <div className="relative z-10 max-w-6xl w-full mx-auto space-y-10">
        {/* Main Standby Status Banner */}
        <div className="bg-canvas-surfaceLow/90 border border-white/[0.09] rounded-2xl p-6 sm:p-8 shadow-2xl relative overflow-hidden backdrop-blur-md">
          {/* Subtle top indicator bar */}
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-amber-500/20 via-brass to-amber-500/20" />

          <div className="flex flex-col md:flex-row items-center md:items-start justify-between gap-6">
            {/* Left: Status Message & Overview */}
            <div className="space-y-3.5 text-center md:text-left max-w-2xl">
              <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-amber-950/40 border border-amber-500/30 text-amber-300 text-xs font-mono">
                <ShieldAlert className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                <span>AWS EC2 Container in Standby (Cost Optimization)</span>
              </div>

              <h1 className="font-serif text-2xl sm:text-3xl font-normal text-white tracking-tight leading-snug">
                Backend Pipeline is Currently in Standby
              </h1>

              <p className="font-sans text-sm text-gray-300 leading-relaxed">
                The LegalRAG neural retrieval pipeline operates over a dual-index cascade (
                <strong className="text-white font-medium">538,079 chunks</strong>, FAISS vector index, BM25, and neural cross-encoder) hosted on an{' '}
                <strong className="text-brass-light font-mono font-normal">AWS EC2 t3.xlarge</strong> instance. To avoid continuous cloud compute costs, the instance is paused when not undergoing active benchmarking.
              </p>

              {/* Rationale Callout */}
              <div className="p-3.5 rounded-xl bg-canvas-base/80 border border-white/[0.06] text-xs font-sans text-gray-300 space-y-1 text-left">
                <div className="flex items-center space-x-1.5 text-brass font-mono text-[11px] uppercase tracking-wider font-medium">
                  <Zap className="w-3.5 h-3.5" />
                  <span>On-Demand Cluster Spin Up</span>
                </div>
                <p className="leading-relaxed text-gray-300 text-[11px]">
                  Evaluating this project or reviewing the retrieval architecture? Reach out on LinkedIn and the admin will immediately spin up the EC2 container for live querying.
                </p>
              </div>
            </div>

            {/* Right: Actions */}
            <div className="flex flex-col sm:flex-row md:flex-col items-stretch gap-3 w-full md:w-72 shrink-0 pt-2">
              {/* Primary LinkedIn Action */}
              <a
                href={linkedinUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center space-x-2.5 px-5 py-3 rounded-xl bg-brass hover:bg-brass-light text-canvas-base font-medium text-xs uppercase tracking-wider transition-all duration-150 shadow-lg hover:shadow-brass/20 cursor-pointer group focus:outline-none focus-visible:ring-2 focus-visible:ring-brass text-center"
              >
                <svg
                  className="w-4 h-4 fill-current transition-transform group-hover:scale-110 shrink-0"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z" />
                </svg>
                <span>Request Container Spin Up</span>
                <ExternalLink className="w-3.5 h-3.5 opacity-70 group-hover:opacity-100 shrink-0" />
              </a>

              {/* Secondary Retry Button */}
              <button
                type="button"
                onClick={handleManualRetry}
                disabled={isRetrying}
                className="inline-flex items-center justify-center space-x-2 px-5 py-3 rounded-xl bg-canvas-surfaceHigh hover:bg-canvas-surfaceHighest border border-white/[0.08] hover:border-brass/30 text-gray-200 hover:text-white text-xs font-mono transition-all disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus-visible:ring-1 focus-visible:ring-brass text-center"
              >
                <RefreshCw
                  className={`w-3.5 h-3.5 text-slateSteel shrink-0 ${
                    isRetrying ? 'animate-spin text-brass' : ''
                  }`}
                />
                <span>{isRetrying ? 'Checking Endpoint...' : 'Re-check /ready Probe'}</span>
              </button>

              {/* Feedback message if retry failed */}
              {retryMessage && (
                <p
                  role="status"
                  aria-live="polite"
                  className="text-center text-xs font-mono text-amber-300/90 animate-fade-in motion-reduce:animate-none"
                >
                  {retryMessage}
                </p>
              )}
            </div>
          </div>

          {/* Quick KPI Strip */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-6 mt-6 border-t border-white/[0.06]">
            <div className="p-3 rounded-xl bg-canvas-base/60 border border-white/[0.04] space-y-1">
              <span className="text-[10px] font-mono text-slateSteel uppercase block">
                Corpus Scale
              </span>
              <span className="text-sm sm:text-base font-serif font-bold text-white block">
                100,000 Cases
              </span>
              <span className="text-[10px] font-mono text-brass-light block">
                24 High Courts
              </span>
            </div>

            <div className="p-3 rounded-xl bg-canvas-base/60 border border-white/[0.04] space-y-1">
              <span className="text-[10px] font-mono text-slateSteel uppercase block">
                Partitioned Chunks
              </span>
              <span className="text-sm sm:text-base font-serif font-bold text-white block">
                538,079 Chunks
              </span>
              <span className="text-[10px] font-mono text-vectorMint block">
                1,200 char window
              </span>
            </div>

            <div className="p-3 rounded-xl bg-canvas-base/60 border border-white/[0.04] space-y-1">
              <span className="text-[10px] font-mono text-slateSteel uppercase block">
                Candidate Recall@50
              </span>
              <span className="text-sm sm:text-base font-serif font-bold text-white block">
                65.79%
              </span>
              <span className="text-[10px] font-mono text-amber-400 block">
                Hybrid-RRF Fusion
              </span>
            </div>

            <div className="p-3 rounded-xl bg-canvas-base/60 border border-white/[0.04] space-y-1">
              <span className="text-[10px] font-mono text-slateSteel uppercase block">
                RAG Success Rate
              </span>
              <span className="text-sm sm:text-base font-serif font-bold text-white block">
                80.2%
              </span>
              <span className="text-[10px] font-mono text-vectorMint block">
                When gold chunk in Top-5
              </span>
            </div>
          </div>
        </div>

        {/* Complete Interactive Architecture & RAG Intelligence Showcase */}
        <ModelArchitectureInfo />
      </div>
    </div>
  );
};
