import React from 'react';
import { Citation } from '../types';

interface GroundedAnswerProps {
  answer: string;
  citations?: Citation[];
  selectedChunkId?: string | null;
  onCitationClick?: (chunkId: string) => void;
}

/**
 * Renders the grounded legal synthesis prose with non-distracting,
 * elegant inline citation badges and clean editorial typography.
 */
export const GroundedAnswer: React.FC<GroundedAnswerProps> = ({
  answer,
  citations = [],
  selectedChunkId,
  onCitationClick,
}) => {
  // Build a lookup map: chunk_id -> { index: number, citation: Citation }
  const citationMap = React.useMemo(() => {
    const map = new Map<string, { index: number; citation: Citation }>();
    citations.forEach((c, idx) => {
      if (c.chunk_id) {
        map.set(c.chunk_id.trim(), { index: idx + 1, citation: c });
      }
    });
    return map;
  }, [citations]);

  // Helper to format inline text with bolding and sleek citation badges
  const renderInlineFormattedText = (text: string, keyPrefix: string): React.ReactNode[] => {
    // Matches [Chunk ID: ...], [Chunk: ...], [Chunk ID ...], or [Chunk ...]
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
          <span key={`${keyPrefix}-txt-${lastIndex}`}>
            {renderBoldAndItalic(precedingText, `${keyPrefix}-b-${lastIndex}`)}
          </span>
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

        // Clean, non-distracting label: e.g. "[1]" or "[1 · CAL]"
        const badgeLabel = citationNum
          ? courtCode
            ? `[${citationNum} · ${courtCode.toUpperCase()}]`
            : `[${citationNum}]`
          : `[Ref]`;

        const tooltipTitle = matched?.citation.title
          ? `[${citationNum}] ${matched.citation.title} (${matched.citation.court_code ? `${matched.citation.court_code.toUpperCase()} High Court` : 'High Court'})\nClick to inspect authority`
          : `Chunk ID: ${chunkId}`;

        elements.push(
          <button
            key={`${keyPrefix}-badge-${matchStart}-${idIdx}`}
            type="button"
            onClick={() => onCitationClick?.(chunkId)}
            className={`inline-flex items-center px-1.5 py-0.5 mx-0.5 my-0 rounded text-[11px] font-mono font-medium transition-all duration-150 cursor-pointer select-none align-baseline focus:outline-none focus-visible:ring-1 focus-visible:ring-brass ${
              isSelected
                ? 'bg-brass text-canvas-base font-bold shadow-sm ring-1 ring-brass scale-[1.03]'
                : 'bg-brass/10 hover:bg-brass/25 text-brass-light hover:text-white border border-brass/25 hover:border-brass/50'
            }`}
            title={tooltipTitle}
            aria-label={`Jump to citation reference ${badgeLabel}`}
          >
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
        <span key={`${keyPrefix}-txt-${lastIndex}`}>
          {renderBoldAndItalic(remainingText, `${keyPrefix}-b-${lastIndex}`)}
        </span>
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

  // Split synthesis answer by line breaks into structured paragraphs and elements
  const lines = answer.split('\n');

  return (
    <div className="space-y-4 text-gray-200 font-sans text-[15px] sm:text-[15.5px] leading-[1.8] antialiased">
      {lines.map((line, idx) => {
        const trimmed = line.trim();

        // Skip empty lines or render subtle spacing
        if (!trimmed) {
          return <div key={`spacer-${idx}`} className="h-1.5" />;
        }

        // Section Headings (### Key Findings or ## Section 482)
        if (trimmed.startsWith('### ') || trimmed.startsWith('## ') || trimmed.startsWith('# ')) {
          const headerText = trimmed.replace(/^#+\s*/, '');
          return (
            <h4
              key={`h-${idx}`}
              className="font-serif text-base sm:text-lg font-medium text-white tracking-tight pt-3 pb-1 border-b border-white/[0.06] flex items-center space-x-2"
            >
              <span className="text-brass font-normal">§</span>
              <span>{headerText}</span>
            </h4>
          );
        }

        // Bullet list items (- , * , •)
        if (trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('• ')) {
          const bulletText = trimmed.replace(/^[-*•]\s*/, '');
          return (
            <div key={`bullet-${idx}`} className="flex items-start space-x-3 pl-1 sm:pl-2">
              <span className="w-1.5 h-1.5 rounded-full bg-brass/70 mt-2.5 shrink-0" />
              <div className="flex-1 text-gray-300">
                {renderInlineFormattedText(bulletText, `line-${idx}`)}
              </div>
            </div>
          );
        }

        // Numbered list items (1. , 2. )
        const numberedMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);
        if (numberedMatch) {
          const num = numberedMatch[1];
          const itemText = numberedMatch[2];
          return (
            <div key={`num-${idx}`} className="flex items-start space-x-3 pl-1 sm:pl-2">
              <span className="font-mono text-xs font-semibold text-brass/90 mt-1 shrink-0 w-4">
                {num}.
              </span>
              <div className="flex-1 text-gray-300">
                {renderInlineFormattedText(itemText, `line-${idx}`)}
              </div>
            </div>
          );
        }

        // Blockquotes (> ...)
        if (trimmed.startsWith('> ')) {
          const quoteText = trimmed.replace(/^>\s*/, '');
          return (
            <blockquote
              key={`quote-${idx}`}
              className="border-l-2 border-brass/50 bg-canvas-surface/70 px-4 py-3 rounded-r text-gray-300 font-serif italic text-sm sm:text-base my-2"
            >
              {renderInlineFormattedText(quoteText, `line-${idx}`)}
            </blockquote>
          );
        }

        // Standard Paragraph
        return (
          <p key={`p-${idx}`} className="text-gray-300">
            {renderInlineFormattedText(trimmed, `line-${idx}`)}
          </p>
        );
      })}
    </div>
  );
};
