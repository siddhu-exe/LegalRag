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
    : 'Indian High Court Jurisdiction';

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
      className={`group relative p-4 sm:p-5 rounded-lg border text-left transition-all duration-200 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-brass ${
        isSelected
          ? 'bg-canvas-surfaceHigh border-brass shadow-[0_0_25px_-5px_rgba(212,175,55,0.25)] ring-1 ring-brass/40'
          : 'bg-canvas-surfaceLow hover:bg-canvas-surfaceHigh/90 border-white/[0.08] hover:border-brass/40'
      }`}
    >
      {/* Active Indicator Bar on left */}
      {isSelected && (
        <div className="absolute left-0 top-3 bottom-3 w-1 bg-brass rounded-r" />
      )}

      {/* Header: Index Seal + Court + Date */}
      <div className="flex items-start justify-between gap-3 mb-2.5">
        <div className="flex items-center space-x-2">
          <div
            className={`w-6 h-6 rounded-sm flex items-center justify-center font-mono text-xs font-bold transition-colors ${
              isSelected
                ? 'bg-brass text-canvas-base shadow-sm'
                : 'bg-brass/10 border border-brass/30 text-brass group-hover:bg-brass/20'
            }`}
          >
            §{index + 1}
          </div>
          <div className="flex items-center space-x-1.5 text-xs font-mono text-slateSteel-light font-medium">
            <Building2 className="w-3.5 h-3.5 text-slateSteel shrink-0" />
            <span className="truncate max-w-[180px] sm:max-w-[220px]" title={courtDisplay}>
              {courtDisplay}
            </span>
          </div>
        </div>

        {citation.decision_date && (
          <div className="flex items-center space-x-1 text-[11px] font-mono text-gray-400 bg-canvas-base/60 px-2 py-0.5 rounded border border-white/[0.04]">
            <Calendar className="w-3 h-3 text-gray-500" />
            <span>{citation.decision_date}</span>
          </div>
        )}
      </div>

      {/* Judgment Title in Editorial Serif */}
      <h4 className="font-serif italic text-sm sm:text-base font-normal text-white group-hover:text-brass-light leading-snug mb-3 line-clamp-2 transition-colors">
        {titleDisplay}
      </h4>

      {/* Metadata & Identifier Pills */}
      <div className="flex flex-wrap items-center gap-2 pt-2.5 border-t border-white/[0.06] text-[11px] font-mono">
        {/* CNR Code with Copy Action */}
        {citation.cnr && (
          <button
            type="button"
            onClick={(e) => handleCopy(e, citation.cnr!, 'cnr')}
            className="inline-flex items-center space-x-1 px-2 py-0.5 rounded bg-canvas-base border border-white/[0.08] hover:border-slateSteel text-gray-300 hover:text-white transition-colors"
            title="Click to copy Case Number Record (CNR)"
          >
            <span className="text-slateSteel">CNR:</span>
            <span className="text-gray-200">{citation.cnr}</span>
            {copiedField === 'cnr' ? (
              <Check className="w-3 h-3 text-vectorMint ml-0.5" />
            ) : (
              <Copy className="w-3 h-3 text-gray-500 hover:text-gray-300 ml-0.5" />
            )}
          </button>
        )}

        {/* Chunk Identifier Badge with Copy Action */}
        <button
          type="button"
          onClick={(e) => handleCopy(e, citation.chunk_id, 'chunk')}
          className="inline-flex items-center space-x-1 px-2 py-0.5 rounded bg-canvas-base border border-white/[0.08] hover:border-brass/40 text-gray-400 hover:text-brass-light transition-colors max-w-[220px]"
          title={`Click to copy Chunk ID: ${citation.chunk_id}`}
        >
          <Hash className="w-3 h-3 text-brass/70 shrink-0" />
          <span className="truncate">{citation.chunk_id}</span>
          {copiedField === 'chunk' ? (
            <Check className="w-3 h-3 text-vectorMint shrink-0 ml-0.5" />
          ) : (
            <Copy className="w-3 h-3 text-gray-500 hover:text-gray-300 shrink-0 ml-0.5" />
          )}
        </button>
      </div>
    </div>
  );
};
