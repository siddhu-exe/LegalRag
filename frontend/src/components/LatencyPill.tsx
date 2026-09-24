import React from 'react';

interface LatencyPillProps {
  retrieveMs?: number;
  rerankMs?: number;
  generateMs?: number;
  totalMs?: number;
}

export const LatencyPill: React.FC<LatencyPillProps> = ({
  retrieveMs = 0,
  rerankMs = 0,
  generateMs = 0,
  totalMs = 0,
}) => {
  return (
    <div className="flex flex-wrap items-center gap-3 font-mono text-xs text-gray-400 bg-canvas-surfaceLow p-2.5 rounded border border-white/10">
      <div className="flex items-center space-x-1.5">
        <span className="text-slateSteel">RETRIEVE</span>
        <span className="text-vectorMint-light">{retrieveMs.toFixed(0)}ms</span>
      </div>
      <span className="text-white/20">|</span>
      <div className="flex items-center space-x-1.5">
        <span className="text-slateSteel">RERANK</span>
        <span className="text-vectorMint-light">{rerankMs.toFixed(0)}ms</span>
      </div>
      <span className="text-white/20">|</span>
      <div className="flex items-center space-x-1.5">
        <span className="text-slateSteel">GENERATE</span>
        <span className="text-vectorMint-light">{generateMs.toFixed(0)}ms</span>
      </div>
      <span className="text-white/20">|</span>
      <div className="flex items-center space-x-1.5 font-medium">
        <span className="text-gray-300">TOTAL</span>
        <span className="text-brass-light">{totalMs.toFixed(0)}ms</span>
      </div>
    </div>
  );
};
