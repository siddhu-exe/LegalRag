import React, { useState } from 'react';
import { QueryResponse } from '../types';
import { GroundedAnswer } from './GroundedAnswer';
import { CitationCard } from './CitationCard';
import { LatencyPill } from './LatencyPill';
import { copyText } from '../clipboard';
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
  const [synthesisCopy, setSynthesisCopy] = useState<'idle' | 'copied' | 'failed'>('idle');
  const [queryCopy, setQueryCopy] = useState<'idle' | 'copied' | 'failed'>('idle');

  // Synchronize selecting citation and scrolling into view
  const handleCitationSelect = (chunkId: string) => {
    setSelectedChunkId(chunkId);
    const element = document.getElementById(`citation-${chunkId}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  };

  const handleCopySynthesis = async () => {
    const textToCopy = `QUESTION:\n${question}\n\nGROUNDED SYNTHESIS:\n${result.answer}\n\nCITATIONS:\n${result.citations
      .map(
        (c, i) =>
          `[${i + 1}] ${c.title || 'Untitled'} (${c.court_code || 'High Court'}) - Chunk ID: ${c.chunk_id}`
      )
      .join('\n')}`;

    const succeeded = await copyText(textToCopy);
    setSynthesisCopy(succeeded ? 'copied' : 'failed');
    setTimeout(() => setSynthesisCopy('idle'), 2500);
  };

  const handleCopyQuery = async () => {
    const succeeded = await copyText(question);
    setQueryCopy(succeeded ? 'copied' : 'failed');
    setTimeout(() => setQueryCopy('idle'), 2500);
  };

  const isErrorStatus = result.status !== 'ok';

  return (
    <div className="relative min-h-[calc(100dvh-4rem)] overflow-hidden">
      {/* Announce copy outcomes to assistive tech without disturbing layout */}
      <p className="sr-only" aria-live="polite">
        {synthesisCopy === 'copied'
          ? 'Synthesis copied to clipboard'
          : synthesisCopy === 'failed'
          ? 'Copying the synthesis failed. Please select the text manually.'
          : ''}
      </p>
      {/* Subtle ambient lighting */}
      <div
        className="absolute top-10 left-1/4 w-[280px] sm:w-[500px] h-[250px] bg-brass/[0.04] rounded-full blur-[140px] pointer-events-none z-0"
        aria-hidden="true"
      />

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-5 pb-16">
        {/* Top Navigation & Action Strip */}
        <div className="flex flex-wrap items-center justify-between gap-3 mb-6 pb-4 border-b border-white/[0.08]">
          <button
            type="button"
            onClick={onNewSearch}
            className="group inline-flex items-center space-x-2 px-3 py-2 sm:py-1.5 min-h-[44px] sm:min-h-0 rounded-lg bg-canvas-surfaceLow hover:bg-canvas-surfaceHigh border border-white/[0.08] hover:border-brass/30 text-xs font-mono text-gray-300 hover:text-white transition-all shadow-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-brass"
          >
            <ArrowLeft className="w-3.5 h-3.5 text-brass group-hover:-translate-x-0.5 transition-transform" />
            <span>New Inquiry</span>
          </button>

          <div className="flex flex-wrap items-center justify-end gap-2.5">
            {/* Status Indicator */}
            {isErrorStatus ? (
              <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-red-950/40 border border-red-500/30 text-xs font-mono text-red-300">
                <AlertTriangle className="w-3 h-3 text-red-400" />
                <span>Notice: {result.status}</span>
              </div>
            ) : (
              <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-canvas-surfaceLow border border-white/[0.06] text-xs font-mono text-slateSteel-light">
                <span className="w-1.5 h-1.5 rounded-full bg-vectorMint shadow-[0_0_6px_#2DD4BF]"></span>
                <span>Grounded Synthesis</span>
              </div>
            )}

            {/* Copy Synthesis Button */}
            <button
              type="button"
              onClick={handleCopySynthesis}
              className="inline-flex items-center justify-center space-x-1.5 px-3 py-2 sm:py-1.5 min-h-[44px] sm:min-h-0 rounded-lg bg-canvas-surfaceLow hover:bg-canvas-surfaceHigh border border-white/[0.08] hover:border-brass/30 text-xs font-mono text-gray-300 hover:text-brass-light transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-brass"
              title="Copy synthesized answer and citations"
            >
              {synthesisCopy === 'copied' ? (
                <>
                  <Check className="w-3.5 h-3.5 text-vectorMint" />
                  <span className="text-vectorMint">Copied!</span>
                </>
              ) : synthesisCopy === 'failed' ? (
                <>
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                  <span className="text-amber-300">Copy failed</span>
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
        <div className="bg-canvas-surfaceLow/90 border border-white/[0.07] rounded-xl p-4 sm:p-5 mb-6 relative shadow-lg">
          <div className="flex items-center justify-between pb-2 mb-2 border-b border-white/[0.05]">
            <div className="flex items-center space-x-1.5">
              <Scale className="w-3.5 h-3.5 text-brass" />
              <span className="font-mono text-[11px] text-slateSteel uppercase tracking-wider font-medium">
                Inquiry Query
              </span>
            </div>
            <button
              type="button"
              onClick={handleCopyQuery}
              className="inline-flex items-center space-x-1 text-[11px] font-mono text-gray-400 hover:text-white transition-colors"
              title="Copy question text"
            >
              {queryCopy === 'copied' ? (
                <span className="text-vectorMint">Copied!</span>
              ) : queryCopy === 'failed' ? (
                <span className="text-amber-300">Copy failed</span>
              ) : (
                <>
                  <Copy className="w-3 h-3" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>

          <h1 className="font-serif italic text-base sm:text-lg text-white leading-relaxed break-words">
            "{question}"
          </h1>

          <div className="mt-3 pt-2 border-t border-white/[0.04] flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-slateSteel">
            <span>Corpus: 100k Judgments (538k Chunks)</span>
            <span className="text-gray-400">
              {result.citations.length} Authorities Cited
            </span>
          </div>
        </div>

        {/* Dual-Pane Grid Layout: Synthesis on Left (7 cols), Citations on Right (5 cols) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start mb-6">
          {/* Left Column: Clean Grounded Synthesis Reading Pane */}
          <div className="lg:col-span-7 bg-canvas-surfaceLow/90 border border-white/[0.08] rounded-xl p-5 sm:p-7 shadow-xl">
            {/* Header */}
            <div className="flex flex-wrap items-center justify-between gap-y-2.5 pb-3.5 mb-5 border-b border-white/[0.07]">
              <div className="flex items-center space-x-2">
                <div className="w-6 h-6 rounded bg-brass/10 border border-brass/25 flex items-center justify-center text-brass font-serif font-bold text-xs">
                  §
                </div>
                <h2 className="font-serif text-base sm:text-lg font-medium text-white tracking-tight">
                  Judicial Precedent Synthesis
                </h2>
              </div>

              <div className="inline-flex items-center space-x-1.5 px-2 py-0.5 rounded bg-canvas-base border border-white/[0.06] text-xs font-mono text-brass-light shrink-0">
                <ShieldCheck className="w-3.5 h-3.5 text-brass" />
                <span>{result.citations.length} Verified Sources</span>
              </div>
            </div>

            {/* Error Callout if generation had issues */}
            {isErrorStatus && (
              <div className="mb-5 p-3.5 rounded-lg bg-red-950/30 border border-red-500/30 text-red-200 text-xs">
                <div className="font-mono font-semibold uppercase tracking-wider text-red-300 mb-1">
                  Notice: {result.status}
                </div>
                <p className="font-sans">
                  The generation provider returned an advisory status. The candidate precedents below represent the nearest retrieved authorities.
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

          {/* Right Column: Citation Authority Cards */}
          <div className="lg:col-span-5 flex flex-col space-y-3">
            {/* Citations Pane Header */}
            <div className="bg-canvas-surfaceLow/90 border border-white/[0.08] rounded-xl p-3.5 sm:p-4 flex flex-wrap items-center justify-between gap-y-2">
              <div className="flex items-center space-x-2">
                <FileText className="w-4 h-4 text-brass" />
                <h2 className="font-mono text-xs uppercase tracking-wider text-white font-medium">
                  Cited Authorities ({result.citations.length})
                </h2>
              </div>

              <span className="text-[11px] font-mono text-slateSteel bg-canvas-base px-2 py-0.5 rounded border border-white/[0.05]">
                Click badge to highlight
              </span>
            </div>

            {/* Scrollable Citation Card List */}
            <div className="space-y-3 lg:max-h-[640px] lg:overflow-y-auto lg:pr-1">
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
                <div className="p-6 bg-canvas-surfaceLow/80 rounded-xl border border-white/[0.07] text-center space-y-2">
                  <Layers className="w-7 h-7 text-slateSteel mx-auto opacity-50" />
                  <p className="text-xs font-mono text-gray-400 font-medium">
                    No Direct Citations Attributed
                  </p>
                  <p className="text-[11px] font-sans text-slateSteel max-w-xs mx-auto">
                    The model did not anchor specific chunk citations in this synthesis.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Bottom Section: Dedicated Pipeline Telemetry & Disclaimer */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7">
            <LatencyPill
              retrieveMs={result.retrieve_ms}
              rerankMs={result.rerank_ms}
              generateMs={result.generate_ms}
              totalMs={result.total_ms}
            />
          </div>

          <div className="lg:col-span-5 flex items-center">
            <div className="w-full p-3 rounded-lg bg-canvas-surfaceLow/60 border border-white/[0.05] flex items-start space-x-2.5 text-[11px] font-mono text-slateSteel">
              <Info className="w-3.5 h-3.5 text-slateSteel shrink-0 mt-0.5" />
              <p className="leading-relaxed">
                Synthesized autonomously over Indian High Court judgments. Citations reflect candidate passages retrieved via BM25 + BGE Dense + Cross-Encoder reranking.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
