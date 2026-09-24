import React, { useState } from 'react';
import { Citation } from '../types';
import { Building2, Calendar, Copy, Check, Hash } from 'lucide-react';

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
  const [copiedField, setCopiedField] = useState<'cnr' | 'chunk' | null>(null);

  const handleCopy = (e: React.MouseEvent, text: string, field: 'cnr' | 'chunk') => {
    e.stopPropagation();
    navigator.clipboard.writeText(text);
    setCopiedField(field);
    setTimeout(() => {
      setCopiedField(null);
    }, 2000);
  };

  const courtDisplay = citation.court_code
    ? `${citation.court_code.toUpperCase()} High Court`
    : 'High Court Jurisdiction';

  const titleDisplay = citation.title || 'Judicial Precedent Record';

  return (
    <div
      id={`citation-${citation.chunk_id}`}
      onClick={() => onSelect?.(citation.chunk_id)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect?.(citation.chunk_id);
        }
      }}
      className={`group relative p-4 rounded-lg border text-left transition-all duration-200 cursor-pointer focus:outline-none focus-visible:ring-1 focus-visible:ring-brass ${
        isSelected
          ? 'bg-canvas-surfaceHigh/90 border-brass shadow-[0_0_20px_-4px_rgba(212,175,55,0.18)]'
          : 'bg-canvas-surfaceLow/80 hover:bg-canvas-surfaceHigh/60 border-white/[0.07] hover:border-brass/30'
      }`}
    >
      {/* Active Accent Pill Indicator on Left */}
      {isSelected && (
        <div className="absolute left-0 top-2.5 bottom-2.5 w-1 bg-brass rounded-r" />
      )}

      {/* Top Metadata Header: Citation Number + Court + Decision Date */}
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center space-x-2 min-w-0">
          <span
            className={`inline-flex items-center justify-center px-1.5 py-0.5 rounded font-mono text-[11px] font-bold tracking-tight transition-colors ${
              isSelected
                ? 'bg-brass text-canvas-base'
                : 'bg-brass/10 text-brass border border-brass/25 group-hover:bg-brass/20'
            }`}
          >
            [{index + 1}]
          </span>

          <div className="flex items-center space-x-1.5 text-xs font-mono text-slateSteel-light truncate">
            <Building2 className="w-3.5 h-3.5 text-slateSteel shrink-0" />
            <span className="truncate" title={courtDisplay}>
              {courtDisplay}
            </span>
          </div>
        </div>

        {citation.decision_date && (
          <div className="flex items-center space-x-1 text-[11px] font-mono text-gray-400 bg-canvas-base/50 px-1.5 py-0.5 rounded border border-white/[0.04] shrink-0">
            <Calendar className="w-3 h-3 text-gray-500 shrink-0" />
            <span>{citation.decision_date}</span>
          </div>
        )}
      </div>

      {/* Case Law Title in Editorial Serif */}
      <h4 className="font-serif italic text-sm font-normal text-white group-hover:text-brass-light leading-snug mb-2.5 line-clamp-2 transition-colors">
        {titleDisplay}
      </h4>

      {/* Bottom Identifier Strip: CNR & Chunk ID */}
      <div className="flex flex-wrap items-center gap-1.5 pt-2 border-t border-white/[0.05] text-[10px] font-mono">
        {citation.cnr && (
          <button
            type="button"
            onClick={(e) => handleCopy(e, citation.cnr!, 'cnr')}
            className="inline-flex items-center space-x-1 px-1.5 py-0.5 rounded bg-canvas-base/60 border border-white/[0.06] hover:border-slateSteel text-gray-400 hover:text-white transition-colors"
            title="Click to copy Case Number Record (CNR)"
          >
            <span className="text-slateSteel">CNR:</span>
            <span className="text-gray-300">{citation.cnr}</span>
            {copiedField === 'cnr' ? (
              <Check className="w-3 h-3 text-vectorMint ml-0.5 shrink-0" />
            ) : (
              <Copy className="w-2.5 h-2.5 text-gray-500 hover:text-gray-300 ml-0.5 shrink-0" />
            )}
          </button>
        )}

        <button
          type="button"
          onClick={(e) => handleCopy(e, citation.chunk_id, 'chunk')}
          className="inline-flex items-center space-x-1 px-1.5 py-0.5 rounded bg-canvas-base/60 border border-white/[0.06] hover:border-brass/30 text-gray-400 hover:text-brass-light transition-colors max-w-[200px]"
          title={`Click to copy Chunk ID: ${citation.chunk_id}`}
        >
          <Hash className="w-2.5 h-2.5 text-brass/60 shrink-0" />
          <span className="truncate">{citation.chunk_id}</span>
          {copiedField === 'chunk' ? (
            <Check className="w-3 h-3 text-vectorMint ml-0.5 shrink-0" />
          ) : (
            <Copy className="w-2.5 h-2.5 text-gray-500 hover:text-gray-300 ml-0.5 shrink-0" />
          )}
        </button>
      </div>
    </div>
  );
};
