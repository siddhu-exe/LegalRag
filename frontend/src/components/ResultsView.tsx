import React, { useState } from 'react';
import { QueryResponse } from '../types';
import { GroundedAnswer } from './GroundedAnswer';
import { CitationCard } from './CitationCard';
import { LatencyPill } from './LatencyPill';

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

  const handleCitationSelect = (chunkId: string) => {
    setSelectedChunkId(chunkId);
    const element = document.getElementById(`citation-${chunkId}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Top action / navigation bar */}
      <div className="flex items-center justify-between mb-6">
        <button
          type="button"
          onClick={onNewSearch}
          className="inline-flex items-center space-x-2 text-xs font-mono text-gray-400 hover:text-white transition-colors"
        >
          <span>←</span>
          <span>New Inquiry</span>
        </button>

        <div className="flex items-center space-x-2 text-xs font-mono text-gray-400">
          <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
          <span>Status: {result.status}</span>
        </div>
      </div>

      {/* Query Banner */}
      <div className="bg-canvas-surfaceLow border border-white/10 rounded-lg p-5 mb-8">
        <div className="text-[10px] font-mono uppercase tracking-widest text-slateSteel mb-1">
          Jurisprudential Query
        </div>
        <h2 className="font-serif text-lg text-white italic">
          "{question}"
        </h2>
      </div>

      {/* Dual Pane Layout: Grounded Answer & Source Citations */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 mb-8">
        {/* Left / Main: Grounded Synthesis Answer */}
        <div className="lg:col-span-7 bg-canvas-surfaceLow border border-white/10 rounded-lg p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-4 mb-4 border-b border-white/5">
              <h3 className="font-serif text-base font-medium text-white flex items-center space-x-2">
                <span className="text-brass">§</span>
                <span>Grounded Synthesis</span>
              </h3>
              <span className="text-[11px] font-mono text-gray-500">
                {result.citations.length} cited authorities
              </span>
            </div>

            <GroundedAnswer
              answer={result.answer}
              selectedChunkId={selectedChunkId}
              onCitationClick={handleCitationSelect}
            />
          </div>

          <div className="mt-8 pt-4 border-t border-white/5">
            <LatencyPill
              retrieveMs={result.retrieve_ms}
              rerankMs={result.rerank_ms}
              generateMs={result.generate_ms}
              totalMs={result.total_ms}
            />
          </div>
        </div>

        {/* Right Pane: Citation Authority Cards */}
        <div className="lg:col-span-5 flex flex-col">
          <div className="flex items-center justify-between pb-3 mb-3 border-b border-white/10">
            <h3 className="font-mono text-xs uppercase tracking-wider text-slateSteel">
              Cited Judgments ({result.citations.length})
            </h3>
            <span className="text-[11px] font-mono text-gray-500">
              Top 5 Considered
            </span>
          </div>

          <div className="space-y-3 overflow-y-auto max-h-[650px] pr-1">
            {result.citations.length > 0 ? (
              result.citations.map((c, i) => (
                <CitationCard
                  key={c.chunk_id || i}
                  citation={c}
                  index={i}
                  isSelected={selectedChunkId === c.chunk_id}
                  onSelect={handleCitationSelect}
                />
              ))
            ) : (
              <div className="p-6 bg-canvas-surfaceLow rounded border border-white/5 text-center text-xs font-mono text-gray-500">
                No citations cited directly in synthesis.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
