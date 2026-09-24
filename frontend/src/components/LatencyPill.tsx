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
  // Ensure valid non-zero total for percentage calculation
  const safeTotal = totalMs > 0 ? totalMs : retrieveMs + rerankMs + generateMs || 1;
  const retrievePct = Math.max(2, Math.min(100, (retrieveMs / safeTotal) * 100));
  const rerankPct = Math.max(2, Math.min(100, (rerankMs / safeTotal) * 100));
  const generatePct = Math.max(2, Math.min(100, (generateMs / safeTotal) * 100));

  return (
    <div className="rounded-lg bg-canvas-surfaceLow/90 border border-white/[0.08] p-3.5 sm:p-4 text-xs font-mono">
      {/* Telemetry Header */}
      <div className="flex items-center justify-between pb-2.5 mb-3 border-b border-white/[0.06]">
        <div className="flex items-center space-x-2 text-slateSteel">
          <Clock className="w-3.5 h-3.5 text-vectorMint" />
          <span className="uppercase tracking-wider text-[11px] font-medium">
            Pipeline Telemetry & Execution Latency
          </span>
        </div>
        <div className="flex items-center space-x-1.5 text-xs">
          <span className="text-gray-400">Total:</span>
          <span className="text-brass font-bold">{totalMs.toFixed(0)}ms</span>
          <span className="text-gray-600 text-[10px]">({(totalMs / 1000).toFixed(2)}s)</span>
        </div>
      </div>

      {/* Segmented Latency Bar */}
      <div className="w-full h-1.5 bg-canvas-base rounded-full overflow-hidden flex mb-3.5 border border-white/[0.04]">
        <div
          style={{ width: `${retrievePct}%` }}
          className="bg-vectorMint transition-all duration-500"
          title={`Retrieval (BM25 + BGE Dense): ${retrieveMs.toFixed(0)}ms (${retrievePct.toFixed(0)}%)`}
        />
        <div
          style={{ width: `${rerankPct}%` }}
          className="bg-slateSteel transition-all duration-500"
          title={`Reranking (MS-MARCO Cross-Encoder): ${rerankMs.toFixed(0)}ms (${rerankPct.toFixed(0)}%)`}
        />
        <div
          style={{ width: `${generatePct}%` }}
          className="bg-brass transition-all duration-500"
          title={`Generation (Groq LLM): ${generateMs.toFixed(0)}ms (${generatePct.toFixed(0)}%)`}
        />
      </div>

      {/* Metric Breakdown Badges */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
        {/* Stage 1: Retrieval */}
        <div className="flex items-center justify-between p-2 rounded bg-canvas-base/80 border border-white/[0.05]">
          <div className="flex items-center space-x-1.5 text-gray-300">
            <Database className="w-3 h-3 text-vectorMint shrink-0" />
            <span className="text-[10px] uppercase tracking-wider text-slateSteel-light">
              1. Retrieval
            </span>
          </div>
          <span className="text-vectorMint font-semibold text-[11px]">
            {retrieveMs.toFixed(0)}ms
          </span>
        </div>

        {/* Stage 2: Rerank */}
        <div className="flex items-center justify-between p-2 rounded bg-canvas-base/80 border border-white/[0.05]">
          <div className="flex items-center space-x-1.5 text-gray-300">
            <Layers className="w-3 h-3 text-slateSteel shrink-0" />
            <span className="text-[10px] uppercase tracking-wider text-slateSteel-light">
              2. Rerank
            </span>
          </div>
          <span className="text-slateSteel-light font-semibold text-[11px]">
            {rerankMs.toFixed(0)}ms
          </span>
        </div>

        {/* Stage 3: Generation */}
        <div className="flex items-center justify-between p-2 rounded bg-canvas-base/80 border border-white/[0.05]">
          <div className="flex items-center space-x-1.5 text-gray-300">
            <Cpu className="w-3 h-3 text-brass shrink-0" />
            <span className="text-[10px] uppercase tracking-wider text-slateSteel-light">
              3. Generation
            </span>
          </div>
          <span className="text-brass-light font-semibold text-[11px]">
            {generateMs.toFixed(0)}ms
          </span>
        </div>
      </div>
    </div>
  );
};
