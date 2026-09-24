import React from 'react';

interface GroundedAnswerProps {
  answer: string;
  onCitationClick?: (chunkId: string) => void;
  selectedChunkId?: string | null;
}

/**
 * Parses grounded answer text and transforms [Chunk ID: ...] into interactive amber badge buttons.
 */
export const GroundedAnswer: React.FC<GroundedAnswerProps> = ({
  answer,
  onCitationClick,
  selectedChunkId,
}) => {
  // Regex to match [Chunk ID: <id>] or similar chunk tags
  const regex = /\[Chunk ID:\s*([^\]]+)\]/gi;

  const renderFormattedText = () => {
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = regex.exec(answer)) !== null) {
      const matchStart = match.index;
      const matchEnd = regex.lastIndex;
      const chunkId = match[1].trim();

      // Push preceding text
      if (matchStart > lastIndex) {
        parts.push(answer.substring(lastIndex, matchStart));
      }

      // Push citation badge
      const isSelected = selectedChunkId === chunkId;
      parts.push(
        <button
          key={`badge-${matchStart}`}
          type="button"
          onClick={() => onCitationClick?.(chunkId)}
          className={`inline-flex items-center px-1.5 py-0.5 mx-1 my-0.5 rounded text-[11px] font-mono border transition-all ${
            isSelected
              ? 'bg-brass text-canvas-base border-brass font-bold'
              : 'bg-brass/10 text-brass-light hover:bg-brass/25 border-brass/30'
          }`}
          title={`Jump to source: ${chunkId}`}
        >
          § {chunkId.slice(0, 16)}...
        </button>
      );

      lastIndex = matchEnd;
    }

    if (lastIndex < answer.length) {
      parts.push(answer.substring(lastIndex));
    }

    return parts;
  };

  return (
    <div className="prose prose-invert max-w-none text-gray-200 text-sm leading-relaxed font-sans whitespace-pre-wrap">
      {renderFormattedText()}
    </div>
  );
};
