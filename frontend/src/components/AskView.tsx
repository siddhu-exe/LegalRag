import React, { useState, useRef, useEffect } from 'react';
import {
  Scale,
  ArrowRight,
  XCircle,
  AlertCircle,
  Layers,
  Database,
  ShieldCheck,
  Clock,
  CheckCircle2,
} from 'lucide-react';

interface AskViewProps {
  onSearch: (question: string) => void;
  isLoading: boolean;
  onCancel?: () => void;
  error?: string | null;
}

interface BenchmarkQuery {
  category: string;
  query: string;
  jurisdiction: string;
}

const BENCHMARK_QUERIES: BenchmarkQuery[] = [
  {
    category: 'Criminal Procedure',
    query:
      'Section 482 CrPC quashing of FIR in matrimonial disputes after amicable settlement between parties',
    jurisdiction: 'Supreme Court & High Courts',
  },
  {
    category: 'Constitutional Law',
    query:
      'Maintainability of writ petition under Article 226 when statutory alternative remedy exists',
    jurisdiction: 'High Court Writ Jurisdiction',
  },
  {
    category: 'Bail Jurisprudence',
    query:
      'Guiding principles and conditions for grant of anticipatory bail in non-bailable economic offences under Section 438 CrPC',
    jurisdiction: 'Criminal Appellate',
  },
  {
    category: 'Commercial Law',
    query:
      'Rebuttal of statutory presumption of legal liability under Section 139 of the Negotiable Instruments Act in cheque dishonour cases',
    jurisdiction: 'Commercial & Special Courts',
  },
  {
    category: 'Arbitration & Conciliation',
    query:
      'Scope of judicial intervention and setting aside arbitral awards under Section 34 of the Arbitration and Conciliation Act for patent illegality',
    jurisdiction: 'Commercial Division',
  },
  {
    category: 'Service Jurisprudence',
    query:
      'Entitlement to compassionate appointment when deceased employee family receives family pension and retiral benefits',
    jurisdiction: 'Administrative & Service Benches',
  },
];

const PIPELINE_STAGES = [
  {
    name: 'Hybrid Retrieval Cascade',
    desc: 'BM25 Sparse + BGE-base Dense over 538k chunks (RRF k=60)',
  },
  {
    name: 'Cross-Encoder Precision Rerank',
    desc: 'MS-MARCO MiniLM scoring top-50 candidates → top-5 context',
  },
  {
    name: 'Grounded LLM Synthesis',
    desc: 'Groq inference with strict citation contracts and anti-hallucination',
  },
];

export const AskView: React.FC<AskViewProps> = ({
  onSearch,
  isLoading,
  onCancel,
  error,
}) => {
  const [question, setQuestion] = useState('');
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const charCount = question.length;
  const isTooShort = charCount > 0 && charCount < 3;
  const isTooLong = charCount > 4000;
  const isValid = charCount >= 3 && charCount <= 4000;

  // Live timer during query execution
  useEffect(() => {
    let timer: number | null = null;
    if (isLoading) {
      setElapsedSeconds(0);
      const startTime = Date.now();
      timer = window.setInterval(() => {
        setElapsedSeconds(Math.floor((Date.now() - startTime) / 100) / 10);
      }, 100);
    } else {
      setElapsedSeconds(0);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isLoading]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (isValid && !isLoading) {
      onSearch(question.trim());
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleBenchmarkSelect = (selectedQuery: string) => {
    setQuestion(selectedQuery);
    if (textareaRef.current) {
      textareaRef.current.focus();
      textareaRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  // Determine which pipeline stage is highlighted based on elapsed seconds
  const currentStageIndex =
    elapsedSeconds < 3.5 ? 0 : elapsedSeconds < 5.5 ? 1 : 2;

  return (
    <div className="relative overflow-hidden">
      {/* Ambient Evidentiary Glow (Non-blocking background) */}
      <div
        className="absolute top-10 left-1/2 -translate-x-1/2 w-[700px] h-[350px] bg-brass/5 rounded-full blur-[140px] pointer-events-none -z-10"
        aria-hidden="true"
      />
      <div
        className="absolute top-80 right-1/4 w-[400px] h-[300px] bg-vectorMint/5 rounded-full blur-[120px] pointer-events-none -z-10"
        aria-hidden="true"
      />

      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 pt-10 sm:pt-14 pb-16">
        {/* Editorial Hero Header */}
        <div className="text-center max-w-3xl mx-auto mb-10">
          {/* Pre-flight System Badge */}
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-canvas-surfaceLow border border-white/[0.08] shadow-sm mb-6">
            <span className="w-2 h-2 rounded-full bg-brass shadow-[0_0_8px_#D4AF37] animate-pulse"></span>
            <span className="font-mono text-[11px] text-slateSteel-light tracking-wider uppercase">
              538,079 Embedded Chunks · Dual-Index Active
            </span>
          </div>

          {/* Main Headline */}
          <h1 className="font-serif text-3xl sm:text-5xl lg:text-5xl text-white tracking-tight leading-[1.15] mb-4">
            Autonomous Legal Reasoning with{' '}
            <span className="italic text-brass font-normal">Verifiable</span> Case Law.
          </h1>

          {/* Subtitle */}
          <p className="font-sans text-sm sm:text-base text-gray-400 max-w-2xl mx-auto leading-relaxed">
            Evidence-grounded judicial retrieval and synthesis over 100,000 Indian High Court judgments.
            Every synthesis is strictly anchored to retrieved chunk citations.
          </p>
        </div>

        {/* Query Console Card */}
        <div className="bg-canvas-surfaceLow border border-white/[0.08] rounded-xl p-5 sm:p-7 shadow-2xl relative transition-all duration-300 focus-within:border-brass/40 focus-within:shadow-[0_0_35px_-5px_rgba(212,175,55,0.12)] mb-12">
          {/* Console Header Bar */}
          <div className="flex items-center justify-between pb-3 mb-4 border-b border-white/[0.06]">
            <div className="flex items-center space-x-2">
              <Scale className="w-4 h-4 text-brass" />
              <span className="font-mono text-xs text-slateSteel uppercase tracking-wider font-medium">
                Judicial Inquiry Contextualizer
              </span>
            </div>

            <div className="flex items-center space-x-2 text-[11px] font-mono text-slateSteel">
              <span className="hidden sm:inline-flex items-center space-x-1 text-vectorMint">
                <span className="w-1.5 h-1.5 rounded-full bg-vectorMint"></span>
                <span>BGE Dense + BM25</span>
              </span>
              <span className="text-gray-600 hidden sm:inline">•</span>
              <span>Top-5 Reranked Context</span>
            </div>
          </div>

          {/* Query Form */}
          <form onSubmit={handleSubmit}>
            <div className="relative">
              <textarea
                ref={textareaRef}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="State your legal proposition or factual inquiry (e.g., Section 482 CrPC quashing in matrimonial disputes, Article 226 maintainability, Section 139 NI Act presumptions)..."
                disabled={isLoading}
                rows={4}
                className="w-full bg-canvas-base border border-white/[0.08] rounded-lg p-4 font-sans text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-brass/60 focus:ring-1 focus:ring-brass/60 transition-all resize-none leading-relaxed disabled:opacity-50"
                aria-label="Legal Inquiry Input"
              />
            </div>

            {/* Bottom Form Action Strip */}
            <div className="mt-4 pt-3 border-t border-white/[0.06] flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              {/* Character & Validation Counter */}
              <div className="flex items-center space-x-2 text-xs font-mono">
                <span
                  className={
                    isTooShort
                      ? 'text-amber-400 font-medium'
                      : isTooLong
                      ? 'text-red-400 font-medium'
                      : 'text-gray-400'
                  }
                >
                  {charCount} / 4,000 characters
                </span>
                {isTooShort && (
                  <span className="text-amber-400/90 text-[11px]">
                    (Minimum 3 characters required)
                  </span>
                )}
                {isTooLong && (
                  <span className="text-red-400 text-[11px]">
                    (Maximum 4,000 character limit exceeded)
                  </span>
                )}
              </div>

              {/* Actions: Cancel + Submit Button */}
              <div className="flex items-center space-x-3 self-end sm:self-auto">
                <span className="hidden md:inline-flex items-center space-x-1 text-[11px] font-mono text-gray-500">
                  <kbd className="px-1.5 py-0.5 rounded bg-canvas-surfaceHigh border border-white/[0.08] text-gray-300">
                    ⌘/Ctrl
                  </kbd>
                  <span>+</span>
                  <kbd className="px-1.5 py-0.5 rounded bg-canvas-surfaceHigh border border-white/[0.08] text-gray-300">
                    Enter
                  </kbd>
                </span>

                {isLoading && onCancel && (
                  <button
                    type="button"
                    onClick={onCancel}
                    className="inline-flex items-center space-x-1.5 px-3.5 py-2 text-xs font-mono text-red-400 hover:text-red-300 bg-red-950/20 hover:bg-red-950/40 border border-red-500/30 rounded-lg transition-colors"
                  >
                    <XCircle className="w-3.5 h-3.5" />
                    <span>Cancel</span>
                  </button>
                )}

                <button
                  type="submit"
                  disabled={!isValid || isLoading}
                  className="group relative inline-flex items-center justify-center space-x-2 px-6 py-2.5 rounded-lg bg-brass hover:bg-brass-light text-canvas-base font-sans font-semibold text-xs uppercase tracking-wider transition-all disabled:opacity-40 disabled:cursor-not-allowed shadow-[0_2px_15px_-3px_rgba(212,175,55,0.3)] active:scale-[0.98]"
                >
                  <span>{isLoading ? 'Synthesizing...' : 'Synthesize Precedents'}</span>
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
                </button>
              </div>
            </div>
          </form>

          {/* Live Multi-Stage Progress Visualizer during query */}
          {isLoading && (
            <div className="mt-5 p-4 rounded-lg bg-canvas-base border border-brass/30 shadow-inner">
              <div className="flex items-center justify-between mb-3 pb-2 border-b border-white/[0.06]">
                <div className="flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-brass animate-ping"></span>
                  <span className="font-mono text-xs text-brass font-medium uppercase tracking-wider">
                    Executing Retrieval & Synthesis Cascade
                  </span>
                </div>
                <div className="flex items-center space-x-1 font-mono text-xs text-vectorMint">
                  <Clock className="w-3.5 h-3.5" />
                  <span>{elapsedSeconds.toFixed(1)}s</span>
                </div>
              </div>

              {/* Progress Steps */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {PIPELINE_STAGES.map((stage, idx) => {
                  const isActive = currentStageIndex === idx;
                  const isCompleted = currentStageIndex > idx;
                  return (
                    <div
                      key={idx}
                      className={`p-2.5 rounded border text-xs font-mono transition-all ${
                        isActive
                          ? 'bg-canvas-surfaceHigh border-brass/50 text-white shadow-sm'
                          : isCompleted
                          ? 'bg-canvas-surface/40 border-vectorMint/30 text-gray-300'
                          : 'bg-canvas-surface/20 border-white/[0.04] text-gray-500'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-semibold flex items-center space-x-1.5">
                          {isCompleted ? (
                            <CheckCircle2 className="w-3.5 h-3.5 text-vectorMint" />
                          ) : (
                            <span
                              className={`w-1.5 h-1.5 rounded-full ${
                                isActive ? 'bg-brass animate-pulse' : 'bg-gray-600'
                              }`}
                            ></span>
                          )}
                          <span className={isActive ? 'text-brass-light' : ''}>
                            {idx + 1}. {stage.name}
                          </span>
                        </span>
                      </div>
                      <p className="text-[11px] text-gray-400 line-clamp-2 leading-tight">
                        {stage.desc}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Granular Error Banner with Diagnostic Context */}
          {error && (
            <div className="mt-5 p-4 rounded-lg bg-red-950/30 border border-red-500/40 flex items-start space-x-3 text-red-200">
              <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
              <div className="text-xs space-y-1">
                <div className="font-mono font-semibold uppercase tracking-wider text-red-300">
                  Query Execution Anomaly
                </div>
                <p className="font-sans leading-relaxed">{error}</p>
                <p className="font-mono text-[11px] text-red-400/80 pt-1">
                  Ensure the selected backend is running and receptive on the target endpoint.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Curated Benchmark Test Vectors Section */}
        <div className="mb-14">
          <div className="flex flex-col sm:flex-row sm:items-baseline justify-between mb-5 pb-2 border-b border-white/[0.06]">
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-sm bg-brass"></span>
              <h2 className="font-mono text-xs uppercase tracking-widest text-slateSteel font-medium">
                01 // Benchmark Evaluation Test Vectors
              </h2>
            </div>
            <span className="font-mono text-[11px] text-gray-500 mt-1 sm:mt-0">
              497-Question Evidence-Grounded Suite
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
            {BENCHMARK_QUERIES.map((item, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleBenchmarkSelect(item.query)}
                disabled={isLoading}
                className="text-left p-4 rounded-lg bg-canvas-surface/60 hover:bg-canvas-surfaceHigh border border-white/[0.06] hover:border-brass/40 transition-all duration-200 group flex flex-col justify-between focus:outline-none focus-visible:ring-2 focus-visible:ring-brass"
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono text-[10px] uppercase tracking-wider px-2 py-0.5 rounded bg-canvas-surfaceHigh text-slateSteel-light border border-white/[0.06] group-hover:border-brass/30 group-hover:text-brass-light transition-colors">
                      {item.category}
                    </span>
                    <span className="text-[10px] font-mono text-gray-500">
                      {item.jurisdiction}
                    </span>
                  </div>
                  <p className="font-serif italic text-xs sm:text-sm text-gray-300 group-hover:text-white leading-relaxed mb-3">
                    "{item.query}"
                  </p>
                </div>
                <div className="pt-2 border-t border-white/[0.04] flex items-center justify-between text-[11px] font-mono text-slateSteel group-hover:text-brass-light transition-colors">
                  <span>Populate Query</span>
                  <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Technical Architecture Strip */}
        <div className="pt-8 border-t border-white/[0.08]">
          <div className="flex items-center space-x-2 mb-4">
            <span className="w-2 h-2 rounded-sm bg-vectorMint"></span>
            <h3 className="font-mono text-xs uppercase tracking-widest text-slateSteel font-medium">
              02 // Production Retrieval Pipeline Architecture
            </h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-lg bg-canvas-surfaceLow border border-white/[0.06]">
              <div className="flex items-center space-x-2 text-brass mb-2">
                <Database className="w-4 h-4" />
                <span className="font-mono text-xs font-medium uppercase tracking-wider text-gray-200">
                  Dual-Index Hybrid RRF
                </span>
              </div>
              <p className="text-xs text-gray-400 font-sans leading-relaxed">
                BM25 sparse ranking with legal regex tokenization combined with 768-dim BGE-base dense FAISS embeddings via Reciprocal Rank Fusion (k=60).
              </p>
            </div>

            <div className="p-4 rounded-lg bg-canvas-surfaceLow border border-white/[0.06]">
              <div className="flex items-center space-x-2 text-slateSteel-light mb-2">
                <Layers className="w-4 h-4" />
                <span className="font-mono text-xs font-medium uppercase tracking-wider text-gray-200">
                  Cross-Encoder Reranker
                </span>
              </div>
              <p className="text-xs text-gray-400 font-sans leading-relaxed">
                Re-scores top-50 hybrid candidates to the top-5 highest scoring passages using MS-MARCO MiniLM cross-encoder before LLM context packing.
              </p>
            </div>

            <div className="p-4 rounded-lg bg-canvas-surfaceLow border border-white/[0.06]">
              <div className="flex items-center space-x-2 text-vectorMint mb-2">
                <ShieldCheck className="w-4 h-4" />
                <span className="font-mono text-xs font-medium uppercase tracking-wider text-gray-200">
                  Grounded Attribution
                </span>
              </div>
              <p className="text-xs text-gray-400 font-sans leading-relaxed">
                Groq inference with anti-hallucination prompt contracts, producing verified <code className="text-brass">[Chunk ID]</code> citation tags.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
