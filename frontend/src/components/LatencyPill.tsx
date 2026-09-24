import React from 'react';
import { Clock, Database, Layers, Cpu } from 'lucide-react';

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
  const safeTotal = totalMs > 0 ? totalMs : retrieveMs + rerankMs + generateMs || 1;
  const retrievePct = Math.max(2, Math.min(100, (retrieveMs / safeTotal) * 100));
  const rerankPct = Math.max(2, Math.min(100, (rerankMs / safeTotal) * 100));
  const generatePct = Math.max(2, Math.min(100, (generateMs / safeTotal) * 100));

  return (
    <div className="rounded-lg bg-canvas-surfaceLow/60 border border-white/[0.06] p-3 text-xs font-mono">
      {/* Telemetry Header */}
      <div className="flex items-center justify-between pb-2 mb-2.5 border-b border-white/[0.05]">
        <div className="flex items-center space-x-2 text-slateSteel">
          <Clock className="w-3.5 h-3.5 text-vectorMint" />
          <span className="uppercase tracking-wider text-[10px] font-medium">
            Execution Telemetry
          </span>
        </div>
        <div className="flex items-center space-x-1.5 text-xs">
          <span className="text-gray-400">Total:</span>
          <span className="text-brass font-bold">{safeTotal.toFixed(0)}ms</span>
          <span className="text-slateSteel text-[11px]">({(safeTotal / 1000).toFixed(2)}s)</span>
        </div>
      </div>

      {/* Segmented Latency Bar */}
      <div className="w-full h-1.5 bg-canvas-base rounded-full overflow-hidden flex mb-2.5 border border-white/[0.03]">
        <div
          style={{ width: `${retrievePct}%` }}
          className="bg-vectorMint/90 transition-all duration-500"
          title={`Retrieval (BM25 + BGE Dense): ${retrieveMs.toFixed(0)}ms (${retrievePct.toFixed(0)}%)`}
        />
        <div
          style={{ width: `${rerankPct}%` }}
          className="bg-slateSteel/90 transition-all duration-500"
          title={`Reranking (MS-MARCO Cross-Encoder): ${rerankMs.toFixed(0)}ms (${rerankPct.toFixed(0)}%)`}
        />
        <div
          style={{ width: `${generatePct}%` }}
          className="bg-brass/90 transition-all duration-500"
          title={`Generation (Groq LLM): ${generateMs.toFixed(0)}ms (${generatePct.toFixed(0)}%)`}
        />
      </div>

      {/* Metric Breakdown Badges */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-[11px]">
        {/* Stage 1: Retrieval */}
        <div className="flex items-center justify-between p-1.5 rounded bg-canvas-base/50 border border-white/[0.04]">
          <div className="flex items-center space-x-1 text-gray-400">
            <Database className="w-3 h-3 text-vectorMint shrink-0" />
            <span className="text-[11px] uppercase text-slateSteel-light">1. Retrieval</span>
          </div>
          <span className="text-vectorMint font-medium">{retrieveMs.toFixed(0)}ms</span>
        </div>

        {/* Stage 2: Rerank */}
        <div className="flex items-center justify-between p-1.5 rounded bg-canvas-base/50 border border-white/[0.04]">
          <div className="flex items-center space-x-1 text-gray-400">
            <Layers className="w-3 h-3 text-slateSteel shrink-0" />
            <span className="text-[11px] uppercase text-slateSteel-light">2. Rerank</span>
          </div>
          <span className="text-slateSteel-light font-medium">{rerankMs.toFixed(0)}ms</span>
        </div>

        {/* Stage 3: Generation */}
        <div className="flex items-center justify-between p-1.5 rounded bg-canvas-base/50 border border-white/[0.04]">
          <div className="flex items-center space-x-1 text-gray-400">
            <Cpu className="w-3 h-3 text-brass shrink-0" />
            <span className="text-[11px] uppercase text-slateSteel-light">3. Groq LLM</span>
          </div>
          <span className="text-brass-light font-medium">{generateMs.toFixed(0)}ms</span>
        </div>
      </div>
    </div>
  );
};
