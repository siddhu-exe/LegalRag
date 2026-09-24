import React from 'react';
import { Citation } from '../types';

interface CitationCardProps {
  citation: Citation;
  index: number;
  isSelected?: boolean;
  onSelect?: (chunkId: string) => void;
}

export const CitationCard: React.FC<CitationCardProps> = ({
  citation,
  index,
  isSelected = false,
  onSelect,
}) => {
  return (
    <div
      id={`citation-${citation.chunk_id}`}
      onClick={() => onSelect?.(citation.chunk_id)}
      className={`p-4 rounded border transition-all duration-200 cursor-pointer ${
        isSelected
          ? 'bg-canvas-surfaceHigh border-brass shadow-[0_0_15px_-3px_rgba(212,175,55,0.2)]'
          : 'bg-canvas-surfaceLow hover:bg-canvas-surfaceHigh border-white/10 hover:border-white/20'
      }`}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center space-x-2">
          <span className="w-5 h-5 rounded-full bg-brass/10 border border-brass/30 text-brass text-[10px] font-mono flex items-center justify-center font-bold">
            {index + 1}
          </span>
          <span className="text-xs font-mono text-slateSteel-light font-medium">
            {citation.court_code || 'High Court'}
          </span>
        </div>
        {citation.decision_date && (
          <span className="text-[11px] font-mono text-gray-500">
            {citation.decision_date}
          </span>
        )}
      </div>

      <h4 className="font-serif text-sm font-medium text-white mb-2 line-clamp-2">
        {citation.title || 'Untitled Judgment Record'}
      </h4>

      <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-white/5 text-[10px] font-mono text-gray-400">
        {citation.cnr && (
          <span className="bg-canvas-base px-1.5 py-0.5 rounded border border-white/5">
            CNR: {citation.cnr}
          </span>
        )}
        <span className="text-gray-500 truncate max-w-[200px]" title={citation.chunk_id}>
          ID: {citation.chunk_id}
        </span>
      </div>
    </div>
  );
};
