import React, { useState } from 'react';

interface AskViewProps {
  onSearch: (question: string) => void;
  isLoading: boolean;
  onCancel?: () => void;
  error?: string | null;
}

const BENCHMARK_QUESTIONS = [
  'Section 482 CrPC quashing of FIR in matrimonial disputes after amicable settlement',
  'Maintainability of writ petition under Article 226 when statutory alternative remedy exists',
  'Conditions and guidelines for grant of anticipatory bail in non-bailable offences',
  'Presumption of legal liability under Section 139 Negotiable Instruments Act in cheque dishonour cases',
];

export const AskView: React.FC<AskViewProps> = ({
  onSearch,
  isLoading,
  onCancel,
  error,
}) => {
  const [question, setQuestion] = useState('');

  const isValidLength = question.trim().length >= 3 && question.length <= 4000;

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (isValidLength && !isLoading) {
      onSearch(question.trim());
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <div className="text-center mb-8">
        <h2 className="text-3xl sm:text-4xl font-serif text-white tracking-tight mb-3">
          Judicial Precedent & Synthesis
        </h2>
        <p className="text-gray-400 text-sm max-w-xl mx-auto">
          Autonomous hybrid retrieval and grounded reasoning engine over 100,000 Indian High Court judgments.
        </p>
      </div>

      {/* Query Form */}
      <div className="bg-canvas-surfaceLow border border-white/10 rounded-lg p-6 mb-8 shadow-xl">
        <form onSubmit={handleSubmit}>
          <div className="relative mb-3">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Formulate your legal inquiry (e.g., maintainability of Article 226 petitions, Section 482 quashing)..."
              disabled={isLoading}
              rows={4}
              className="w-full bg-canvas-base border border-white/10 rounded p-4 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-brass focus:ring-1 focus:ring-brass transition-all resize-none"
            />
          </div>

          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2">
            <div className="text-xs font-mono text-gray-500">
              {question.length}/4000 characters
              {question.length > 0 && question.length < 3 && (
                <span className="text-amber-400 ml-2">(minimum 3 chars)</span>
              )}
            </div>

            <div className="flex items-center space-x-3 w-full sm:w-auto">
              {isLoading ? (
                <button
                  type="button"
                  onClick={onCancel}
                  className="px-4 py-2 text-xs font-mono text-red-400 hover:text-red-300 border border-red-500/30 rounded transition-colors"
                >
                  Cancel Query
                </button>
              ) : null}

              <button
                type="submit"
                disabled={!isValidLength || isLoading}
                className="w-full sm:w-auto px-6 py-2.5 bg-brass hover:bg-brass-light text-canvas-base font-semibold text-sm rounded shadow transition-all disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Analyzing Jurisprudence...' : 'Analyze Jurisprudence'}
              </button>
            </div>
          </div>
        </form>

        {error && (
          <div className="mt-4 p-3 bg-red-950/40 border border-red-500/30 rounded text-red-300 text-xs font-mono">
            {error}
          </div>
        )}
      </div>

      {/* Curated Benchmark Questions */}
      <div>
        <h3 className="text-xs font-mono uppercase tracking-widest text-slateSteel mb-3">
          Benchmark Questions
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {BENCHMARK_QUESTIONS.map((q, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => setQuestion(q)}
              disabled={isLoading}
              className="text-left p-3.5 bg-canvas-surface/60 hover:bg-canvas-surfaceHigh border border-white/5 hover:border-white/15 rounded text-xs text-gray-300 hover:text-white transition-all group"
            >
              <span className="font-serif italic text-gray-400 group-hover:text-brass-light block">
                "{q}"
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
