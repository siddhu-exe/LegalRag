import React, { useState } from 'react';
import { Header } from './components/Header';
import { AskView } from './components/AskView';
import { ResultsView } from './components/ResultsView';
import { ActiveView, QueryResponse } from './types';
import { queryLegalRag } from './api';

export const App: React.FC = () => {
  const [view, setView] = useState<ActiveView>('ask');
  const [question, setQuestion] = useState<string>('');
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  const handleSearch = async (queryText: string) => {
    setIsLoading(true);
    setError(null);
    setQuestion(queryText);

    const controller = new AbortController();
    setAbortController(controller);

    try {
      const data = await queryLegalRag({ question: queryText }, controller.signal);
      setResult(data);
      setView('results');
    } catch (err: unknown) {
      if (err instanceof Error) {
        if (err.name === 'AbortError') {
          setError('Inquiry cancelled by user.');
        } else {
          setError(err.message);
        }
      } else {
        setError('An unexpected error occurred while processing the judicial query.');
      }
    } finally {
      setIsLoading(false);
      setAbortController(null);
    }
  };

  const handleCancel = () => {
    if (abortController) {
      abortController.abort();
      setAbortController(null);
      setIsLoading(false);
    }
  };

  const handleNewSearch = () => {
    setView('ask');
    setError(null);
  };

  return (
    <div className="min-h-screen bg-canvas-base flex flex-col font-sans text-gray-200 antialiased">
      <Header onNewSearch={handleNewSearch} />

      <main className="flex-1">
        {view === 'ask' ? (
          <AskView
            onSearch={handleSearch}
            isLoading={isLoading}
            onCancel={handleCancel}
            error={error}
          />
        ) : result ? (
          <ResultsView
            question={question}
            result={result}
            onNewSearch={handleNewSearch}
          />
        ) : null}
      </main>

      {/* Editorial Footer */}
      <footer className="border-t border-white/[0.06] bg-canvas-surface/40 py-6 text-center text-xs font-mono text-gray-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>LegalRAG · Autonomous Judicial Retrieval Engine</span>
          <span className="text-gray-600">
            100,000 High Court Judgments · 538,079 Chunks · BGE + BM25 + Cross-Encoder
          </span>
        </div>
      </footer>
    </div>
  );
};

export default App;
