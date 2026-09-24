import React from 'react';
import { Citation } from '../types';
import { ShieldCheck } from 'lucide-react';

interface GroundedAnswerProps {
  answer: string;
  citations?: Citation[];
  selectedChunkId?: string | null;
  onCitationClick?: (chunkId: string) => void;
}

/**
 * Parses and renders the grounded synthesis with interactive evidentiary citation badges.
 */
export const GroundedAnswer: React.FC<GroundedAnswerProps> = ({
  answer,
  citations = [],
  selectedChunkId,
  onCitationClick,
}) => {
  // Build a lookup map of chunk_id -> citation index (1-based) and citation object
  const citationMap = React.useMemo(() => {
    const map = new Map<string, { index: number; citation: Citation }>();
    citations.forEach((c, idx) => {
      if (c.chunk_id) {
        map.set(c.chunk_id.trim(), { index: idx + 1, citation: c });
      }
    });
    return map;
  }, [citations]);

  // Helper to format inline text with bolding and citation badges
  const renderInlineFormattedText = (text: string, keyPrefix: string): React.ReactNode[] => {
    // Regex matches [Chunk ID: ...] or [Chunk: ...] or [Chunk ID ...]
    const citationRegex = /\[(?:Chunk(?:\s*ID)?[:\s]+)([^\]]+)\]/gi;
    const elements: React.ReactNode[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = citationRegex.exec(text)) !== null) {
      const matchStart = match.index;
      const matchEnd = citationRegex.lastIndex;
      const rawChunkString = match[1].trim();

      // Push text before the match
      if (matchStart > lastIndex) {
        const precedingText = text.substring(lastIndex, matchStart);
        elements.push(
          <span key={`${keyPrefix}-txt-${lastIndex}`}>{renderBoldAndItalic(precedingText, `${keyPrefix}-b-${lastIndex}`)}</span>
        );
      }

      // Handle multiple comma-separated chunk IDs if present in a single tag
      const rawIds = rawChunkString.split(/,\s*/);

      rawIds.forEach((rawId, idIdx) => {
        const chunkId = rawId.trim();
        const matched = citationMap.get(chunkId);
        const isSelected = selectedChunkId === chunkId;
        const citationNum = matched ? matched.index : null;
        const courtCode = matched?.citation.court_code;

        // Label display: e.g. "§1 CAL" or "§1" or shortened chunk
        const badgeLabel = citationNum
          ? `§${citationNum}${courtCode ? ` ${courtCode.toUpperCase()}` : ''}`
          : `§ ${chunkId.slice(0, 14)}…`;

        const tooltipTitle = matched?.citation.title
          ? `Citation §${citationNum}: ${matched.citation.title} (${matched.citation.court_code || 'High Court'})`
          : `Chunk ID: ${chunkId}`;

        elements.push(
          <button
            key={`${keyPrefix}-badge-${matchStart}-${idIdx}`}
            type="button"
            onClick={() => onCitationClick?.(chunkId)}
            className={`inline-flex items-center space-x-1 px-2 py-0.5 mx-1 my-0.5 rounded text-[11px] font-mono font-medium border transition-all duration-200 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-brass ${
              isSelected
                ? 'bg-brass text-canvas-base border-brass font-bold shadow-[0_0_15px_-2px_rgba(212,175,55,0.4)] scale-105'
                : 'bg-brass/10 text-brass-light hover:bg-brass/25 hover:text-white border-brass/30 hover:border-brass/60'
            }`}
            title={tooltipTitle}
            aria-label={`Jump to citation reference ${badgeLabel}`}
          >
            <ShieldCheck className={`w-3 h-3 ${isSelected ? 'text-canvas-base' : 'text-brass'}`} />
            <span>{badgeLabel}</span>
          </button>
        );
      });

      lastIndex = matchEnd;
    }

    // Push remaining text
    if (lastIndex < text.length) {
      const remainingText = text.substring(lastIndex);
      elements.push(
        <span key={`${keyPrefix}-txt-${lastIndex}`}>{renderBoldAndItalic(remainingText, `${keyPrefix}-b-${lastIndex}`)}</span>
      );
    }

    return elements;
  };

  // Helper to parse basic markdown bold (**text**) and italic (*text*)
  const renderBoldAndItalic = (text: string, keyPrefix: string): React.ReactNode => {
    // Split on **bold**
    const boldParts = text.split(/(\*\*[^*]+\*\*)/g);
    return boldParts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        const inner = part.slice(2, -2);
        return (
          <strong key={`${keyPrefix}-bold-${i}`} className="font-semibold text-white">
            {inner}
          </strong>
        );
      }
      return part;
    });
  };

  // Split synthesis answer by line breaks into structured paragraphs and bullet items
  const lines = answer.split('\n');

  return (
    <div className="space-y-4 text-gray-200 font-sans text-sm sm:text-[15px] leading-relaxed">
      {lines.map((line, idx) => {
        const trimmed = line.trim();

        // Skip empty lines or render subtle spacing
        if (!trimmed) {
          return <div key={`spacer-${idx}`} className="h-1" />;
        }

        // Heading 3 / Section Marker (e.g., ### Key Findings or ## Section 482)
        if (trimmed.startsWith('### ') || trimmed.startsWith('## ')) {
          const headerText = trimmed.replace(/^#+\s*/, '');
          return (
            <h4
              key={`h-${idx}`}
              className="font-serif text-base sm:text-lg font-medium text-white tracking-tight pt-2 border-b border-white/[0.06] pb-1.5 flex items-center space-x-2"
            >
              <span className="text-brass">§</span>
              <span>{headerText}</span>
            </h4>
          );
        }

        // Bullet lists
        if (trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('• ')) {
          const bulletText = trimmed.replace(/^[-*•]\s*/, '');
          return (
            <div key={`bullet-${idx}`} className="flex items-start space-x-2.5 pl-2 sm:pl-3">
              <span className="w-1.5 h-1.5 rounded-full bg-brass/80 mt-2 shrink-0"></span>
              <div className="flex-1 text-gray-300">
                {renderInlineFormattedText(bulletText, `line-${idx}`)}
              </div>
            </div>
          );
        }

        // Numbered list items (e.g. 1. , 2. )
        const numberedMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);
        if (numberedMatch) {
          const num = numberedMatch[1];
          const itemText = numberedMatch[2];
          return (
            <div key={`num-${idx}`} className="flex items-start space-x-2.5 pl-2 sm:pl-3">
              <span className="font-mono text-xs font-semibold text-brass mt-1 shrink-0">
                {num}.
              </span>
              <div className="flex-1 text-gray-300">
                {renderInlineFormattedText(itemText, `line-${idx}`)}
              </div>
            </div>
          );
        }

        // Blockquotes (e.g., > ...)
        if (trimmed.startsWith('> ')) {
          const quoteText = trimmed.replace(/^>\s*/, '');
          return (
            <blockquote
              key={`quote-${idx}`}
              className="border-l-2 border-brass/60 bg-canvas-surface/60 px-4 py-2.5 rounded-r text-gray-300 font-serif italic text-sm sm:text-base my-2"
            >
              {renderInlineFormattedText(quoteText, `line-${idx}`)}
            </blockquote>
          );
        }

        // Standard Paragraph
        return (
          <p key={`p-${idx}`} className="text-gray-300 leading-relaxed">
            {renderInlineFormattedText(trimmed, `line-${idx}`)}
          </p>
        );
      })}
    </div>
  );
};
