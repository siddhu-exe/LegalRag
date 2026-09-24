import React, { useState } from 'react';
import { QueryResponse } from '../types';
import { GroundedAnswer } from './GroundedAnswer';
import { CitationCard } from './CitationCard';
import { LatencyPill } from './LatencyPill';
import {
  ArrowLeft,
  Copy,
  Check,
  Scale,
  ShieldCheck,
  AlertTriangle,
  FileText,
  Layers,
  Info,
} from 'lucide-react';

interface ResultsViewProps {
  question: string;
  result: QueryResponse;
  onNewSearch: () => void;
}

export const ResultsView: React.FC<ResultsViewProps> = ({
  question,
  result,
  onNewSearch,
}) => {
  const [selectedChunkId, setSelectedChunkId] = useState<string | null>(null);
  const [copiedSynthesis, setCopiedSynthesis] = useState(false);
  const [copiedQuery, setCopiedQuery] = useState(false);

  // Synchronize selecting citation and scrolling into view
  const handleCitationSelect = (chunkId: string) => {
    setSelectedChunkId(chunkId);
    const element = document.getElementById(`citation-${chunkId}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  };

  const handleCopySynthesis = () => {
    const textToCopy = `QUESTION:\n${question}\n\nGROUNDED SYNTHESIS:\n${result.answer}\n\nCITATIONS:\n${result.citations
      .map(
        (c, i) =>
          `[${i + 1}] ${c.title || 'Untitled'} (${c.court_code || 'High Court'}) - Chunk ID: ${c.chunk_id}`
      )
      .join('\n')}`;

    navigator.clipboard.writeText(textToCopy);
    setCopiedSynthesis(true);
    setTimeout(() => setCopiedSynthesis(false), 2000);
  };

  const handleCopyQuery = () => {
    navigator.clipboard.writeText(question);
    setCopiedQuery(true);
    setTimeout(() => setCopiedQuery(false), 2000);
  };

  const isErrorStatus = result.status !== 'ok';

  return (
    <div className="relative overflow-hidden min-h-[calc(100vh-4rem)]">
      {/* Ambient Evidentiary Glow (Non-blocking background) */}
      <div
        className="absolute top-20 left-1/3 w-[600px] h-[300px] bg-brass/5 rounded-full blur-[140px] pointer-events-none -z-10"
        aria-hidden="true"
      />
      <div
        className="absolute top-96 right-10 w-[400px] h-[300px] bg-vectorMint/5 rounded-full blur-[120px] pointer-events-none -z-10"
        aria-hidden="true"
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6 pb-16">
        {/* Top Navigation & Action Strip */}
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6 pb-4 border-b border-white/[0.08]">
          <button
            type="button"
            onClick={onNewSearch}
            className="group inline-flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-canvas-surfaceLow hover:bg-canvas-surfaceHigh border border-white/[0.08] hover:border-brass/40 text-xs font-mono text-gray-300 hover:text-white transition-all shadow-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-brass"
          >
            <ArrowLeft className="w-3.5 h-3.5 text-brass group-hover:-translate-x-0.5 transition-transform" />
            <span>Return to Inquiry Console</span>
          </button>

          <div className="flex items-center space-x-2.5">
            {/* Status Indicator */}
            {isErrorStatus ? (
              <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-red-950/40 border border-red-500/30 text-xs font-mono text-red-300">
                <AlertTriangle className="w-3 h-3 text-red-400" />
                <span>Status: {result.status}</span>
              </div>
            ) : (
              <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-canvas-surfaceLow border border-white/[0.08] text-xs font-mono text-slateSteel-light">
                <span className="w-2 h-2 rounded-full bg-vectorMint shadow-[0_0_8px_#2DD4BF]"></span>
                <span>Verified Grounded Synthesis</span>
              </div>
            )}

            {/* Copy Actions */}
            <button
              type="button"
              onClick={handleCopySynthesis}
              className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-canvas-surfaceLow hover:bg-canvas-surfaceHigh border border-white/[0.08] hover:border-brass/40 text-xs font-mono text-gray-300 hover:text-brass-light transition-all"
              title="Copy synthesized answer and citations to clipboard"
            >
              {copiedSynthesis ? (
                <>
                  <Check className="w-3.5 h-3.5 text-vectorMint" />
                  <span className="text-vectorMint">Copied Synthesis!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5 text-gray-400" />
                  <span>Copy Synthesis</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Inquiry Question Banner */}
        <div className="bg-canvas-surfaceLow border border-white/[0.08] rounded-xl p-5 sm:p-6 mb-8 relative transition-all shadow-xl">
          <div className="flex items-center justify-between pb-2 mb-3 border-b border-white/[0.06]">
            <div className="flex items-center space-x-2">
              <Scale className="w-4 h-4 text-brass" />
              <span className="font-mono text-xs text-slateSteel uppercase tracking-wider font-medium">
                Jurisprudential Inquiry Record
              </span>
            </div>
            <button
              type="button"
              onClick={handleCopyQuery}
              className="inline-flex items-center space-x-1 text-[11px] font-mono text-gray-400 hover:text-white transition-colors"
              title="Copy question text"
            >
              {copiedQuery ? (
                <span className="text-vectorMint">Copied!</span>
              ) : (
                <>
                  <Copy className="w-3 h-3" />
                  <span>Copy Query</span>
                </>
              )}
            </button>
          </div>

          <h2 className="font-serif italic text-base sm:text-xl text-white leading-relaxed">
            "{question}"
          </h2>

          <div className="mt-4 pt-3 border-t border-white/[0.04] flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-slateSteel">
            <span>Dual-Index Hybrid Retrieval Cascade (538,079 Chunks)</span>
            <span className="text-gray-500">
              {result.retrieved_chunk_ids?.length || 5} Candidate Passages Reranked → {result.citations.length} Authorities Cited
            </span>
          </div>
        </div>

        {/* Dual-Pane Grid Layout: Synthesis on Left (7 cols), Citations on Right (5 cols) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start mb-8">
          {/* Left Column: Grounded Synthesis Answer */}
          <div className="lg:col-span-7 bg-canvas-surfaceLow border border-white/[0.08] rounded-xl p-6 sm:p-8 shadow-2xl flex flex-col justify-between">
            <div>
              {/* Grounded Synthesis Header */}
              <div className="flex flex-wrap items-center justify-between gap-3 pb-4 mb-6 border-b border-white/[0.08]">
                <div className="flex items-center space-x-2.5">
                  <div className="w-7 h-7 rounded bg-brass/10 border border-brass/30 flex items-center justify-center text-brass font-serif font-bold text-sm">
                    §
                  </div>
                  <div>
                    <h3 className="font-serif text-lg font-medium text-white tracking-tight">
                      Judicial Precedent Synthesis
                    </h3>
                    <p className="text-[11px] font-mono text-slateSteel">
                      Grounded LLM Inference · Groq Provider
                    </p>
                  </div>
                </div>

                <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded bg-canvas-base border border-white/[0.06] text-xs font-mono text-brass-light">
                  <ShieldCheck className="w-3.5 h-3.5 text-brass" />
                  <span>{result.citations.length} Cited Authorities</span>
                </div>
              </div>

              {/* Error Callout if generation had issues */}
              {isErrorStatus && (
                <div className="mb-6 p-4 rounded-lg bg-red-950/30 border border-red-500/30 text-red-200 text-xs">
                  <div className="font-mono font-semibold uppercase tracking-wider text-red-300 mb-1">
                    Pipeline Generation Notice: {result.status}
                  </div>
                  <p className="font-sans">
                    The generation stage encountered an anomaly on the provider. Retrieved passages below reflect the nearest candidate precedents.
                  </p>
                </div>
              )}

              {/* Main Grounded Prose Body */}
              <GroundedAnswer
                answer={result.answer}
                citations={result.citations}
                selectedChunkId={selectedChunkId}
                onCitationClick={handleCitationSelect}
              />
            </div>

            {/* Bottom Section: Telemetry & Disclaimer */}
            <div className="mt-8 pt-6 border-t border-white/[0.08] space-y-4">
              {/* Latency Breakdown Pill */}
              <LatencyPill
                retrieveMs={result.retrieve_ms}
                rerankMs={result.rerank_ms}
                generateMs={result.generate_ms}
                totalMs={result.total_ms}
              />

              {/* Academic & Portfolio Context Disclaimer */}
              <div className="p-3 rounded bg-canvas-base/60 border border-white/[0.04] flex items-start space-x-2 text-[11px] font-mono text-gray-500">
                <Info className="w-3.5 h-3.5 text-slateSteel shrink-0 mt-0.5" />
                <p className="leading-normal">
                  Synthesized autonomously over 100,000 Indian High Court judgments. Citations reflect top candidate passages selected via BM25 + BGE Dense + Cross-Encoder reranking.
                </p>
              </div>
            </div>
          </div>

          {/* Right Column: Citation Authority Cards & Context Inspector */}
          <div className="lg:col-span-5 flex flex-col space-y-4">
            {/* Citations Pane Header */}
            <div className="bg-canvas-surfaceLow border border-white/[0.08] rounded-xl p-4 sm:p-5 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <FileText className="w-4 h-4 text-brass" />
                <div>
                  <h3 className="font-mono text-xs uppercase tracking-wider text-white font-medium">
                    Cited Authorities ({result.citations.length})
                  </h3>
                  <p className="text-[10px] font-mono text-slateSteel">
                    Interactive Precedent Grounding
                  </p>
                </div>
              </div>

              <span className="text-[11px] font-mono text-gray-400 bg-canvas-base px-2.5 py-1 rounded border border-white/[0.06]">
                Top-5 Reranked
              </span>
            </div>

            {/* Scrollable Citation Card List */}
            <div className="space-y-3.5 max-h-[720px] overflow-y-auto pr-1">
              {result.citations.length > 0 ? (
                result.citations.map((citation, index) => (
                  <CitationCard
                    key={citation.chunk_id || index}
                    citation={citation}
                    index={index}
                    isSelected={selectedChunkId === citation.chunk_id}
                    onSelect={handleCitationSelect}
                  />
                ))
              ) : (
                <div className="p-8 bg-canvas-surfaceLow rounded-xl border border-white/[0.08] text-center space-y-2">
                  <Layers className="w-8 h-8 text-slateSteel mx-auto opacity-50" />
                  <p className="text-xs font-mono text-gray-400 font-medium">
                    No Direct Citations Attributed
                  </p>
                  <p className="text-[11px] font-sans text-gray-500 max-w-xs mx-auto">
                    The synthesis model did not explicitly anchor specific chunk markers in the response body.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
