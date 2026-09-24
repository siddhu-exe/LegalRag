import React, { useState } from 'react';
import {
  ServerOff,
  RefreshCw,
  ExternalLink,
  Cpu,
  Database,
  Layers,
  ShieldAlert,
  Zap,
  CheckCircle2,
} from 'lucide-react';

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
    <div className="relative overflow-hidden min-h-[calc(100dvh-8rem)] flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      {/* Ambient background glow */}
      <div
        className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[300px] sm:w-[600px] h-[300px] bg-brass/5 rounded-full blur-[160px] pointer-events-none z-0"
        aria-hidden="true"
      />

      <div className="relative z-10 max-w-2xl w-full mx-auto space-y-8">
        {/* Main Status Container */}
        <div className="bg-canvas-surfaceLow/90 border border-white/[0.09] rounded-2xl p-6 sm:p-10 shadow-2xl relative overflow-hidden backdrop-blur-md">
          {/* Subtle top indicator bar */}
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-amber-500/20 via-brass to-amber-500/20" />

          {/* Header & Icon */}
          <div className="flex flex-col items-center text-center space-y-4">
            <div className="relative">
              <div className="w-16 h-16 rounded-2xl bg-canvas-surfaceHigh/80 border border-brass/30 flex items-center justify-center text-brass shadow-[0_0_30px_-5px_rgba(212,175,55,0.2)]">
                <ServerOff className="w-8 h-8 text-brass" />
              </div>
              <span className="absolute -bottom-1 -right-1 flex h-4 w-4">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-4 w-4 bg-amber-500"></span>
              </span>
            </div>

            <div className="space-y-2">
              <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-amber-950/40 border border-amber-500/30 text-amber-300 text-xs font-mono">
                <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
                <span>AWS EC2 Container in Standby</span>
              </div>

              <h1 className="font-serif text-2xl sm:text-3xl font-normal text-white tracking-tight">
                Backend Pipeline is Currently Paused
              </h1>
            </div>

            <p className="font-sans text-sm sm:text-[15px] text-gray-300 leading-relaxed max-w-lg">
              The LegalRAG neural retrieval pipeline operates over a heavy dual-index cascade
              (<strong className="text-white font-medium">538,079 chunks</strong>, FAISS vector index, BM25, and neural cross-encoder)
              hosted on an <strong className="text-brass-light font-mono font-normal">AWS EC2 t4.xlarge</strong> container instance.
            </p>

            <div className="p-4 rounded-xl bg-canvas-base/80 border border-white/[0.06] text-left text-xs font-sans text-gray-300 space-y-2 w-full">
              <div className="flex items-center space-x-2 text-brass font-mono text-[11px] uppercase tracking-wider font-medium">
                <Zap className="w-3.5 h-3.5" />
                <span>Why is the backend offline?</span>
              </div>
              <p className="leading-relaxed">
                To prevent recurring cloud compute costs on AWS, the container instance is kept in standby when not actively benchmarking or undergoing scheduled evaluation.
              </p>
              <p className="leading-relaxed text-gray-400">
                If you are reviewing this portfolio project or evaluating the retrieval engine, please reach out on LinkedIn so the admin can spin up the EC2 container on demand.
              </p>
            </div>
          </div>

          {/* Action CTAs */}
          <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-3.5">
            {/* Primary LinkedIn Action */}
            <a
              href={linkedinUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="w-full sm:w-auto inline-flex items-center justify-center space-x-2.5 px-6 py-3 rounded-xl bg-brass hover:bg-brass-light text-canvas-base font-medium text-sm transition-all duration-150 shadow-lg hover:shadow-brass/20 cursor-pointer group focus:outline-none focus-visible:ring-2 focus-visible:ring-brass"
            >
              {/* LinkedIn SVG Icon */}
              <svg
                className="w-4 h-4 fill-current transition-transform group-hover:scale-110"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z" />
              </svg>
              <span>Contact Admin on LinkedIn to Turn On</span>
              <ExternalLink className="w-3.5 h-3.5 opacity-70 group-hover:opacity-100" />
            </a>

            {/* Secondary Retry Button */}
            <button
              type="button"
              onClick={handleManualRetry}
              disabled={isRetrying}
              className="w-full sm:w-auto inline-flex items-center justify-center space-x-2 px-5 py-3 rounded-xl bg-canvas-surfaceHigh hover:bg-canvas-surfaceHighest border border-white/[0.08] hover:border-brass/30 text-gray-200 hover:text-white text-sm font-mono transition-all disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus-visible:ring-1 focus-visible:ring-brass"
            >
              <RefreshCw
                className={`w-4 h-4 text-slateSteel ${
                  isRetrying ? 'animate-spin text-brass' : ''
                }`}
              />
              <span>{isRetrying ? 'Checking Backend...' : 'Re-check Status'}</span>
            </button>
          </div>

          {/* Feedback message if retry failed */}
          {retryMessage && (
            <p
              role="status"
              aria-live="polite"
              className="mt-4 text-center text-xs font-mono text-amber-300/90 animate-fade-in motion-reduce:animate-none"
            >
              {retryMessage}
            </p>
          )}
        </div>

        {/* Technical Architecture Specs Card */}
        <div className="bg-canvas-surfaceLow/60 border border-white/[0.05] rounded-xl p-5 text-xs font-mono text-gray-400 space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-white/[0.05] text-slateSteel">
            <span className="uppercase tracking-wider text-[11px] font-medium flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5 text-brass" />
              <span>Target Infrastructure Profile</span>
            </span>
            <span className="text-[11px] bg-canvas-base px-2 py-0.5 rounded border border-white/[0.05] text-slateSteel-light shrink-0">
              AWS EC2
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
            <div className="flex items-start space-x-2">
              <Layers className="w-4 h-4 text-slateSteel mt-0.5 shrink-0" />
              <div>
                <span className="text-gray-300 block font-medium">Compute</span>
                <span className="text-[11px] text-slateSteel">t4.xlarge (4 vCPUs · 16 GiB)</span>
              </div>
            </div>

            <div className="flex items-start space-x-2">
              <Database className="w-4 h-4 text-slateSteel mt-0.5 shrink-0" />
              <div>
                <span className="text-gray-300 block font-medium">Corpus Scale</span>
                <span className="text-[11px] text-slateSteel">538,079 Clean Chunks</span>
              </div>
            </div>

            <div className="flex items-start space-x-2">
              <CheckCircle2 className="w-4 h-4 text-vectorMint mt-0.5 shrink-0" />
              <div>
                <span className="text-gray-300 block font-medium">Pipeline</span>
                <span className="text-[11px] text-slateSteel">BM25 + FAISS + Cross-Encoder</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
