import React, { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { AskView } from './components/AskView';
import { ResultsView } from './components/ResultsView';
import { BackendOfflineView } from './components/BackendOfflineView';
import { ActiveView, QueryResponse, BackendStatus } from './types';
import { queryLegalRag, checkBackendReady } from './api';
import { Loader2, Server } from 'lucide-react';

export const App: React.FC = () => {
  const [backendStatus, setBackendStatus] = useState<BackendStatus>('checking');
  const [isRetryingReadiness, setIsRetryingReadiness] = useState<boolean>(false);
  const [view, setView] = useState<ActiveView>('ask');
  const [question, setQuestion] = useState<string>('');
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  const verifyReadiness = useCallback(async () => {
    try {
      const isReady = await checkBackendReady();
      setBackendStatus(isReady ? 'ready' : 'offline');
      return isReady;
    } catch {
      setBackendStatus('offline');
      return false;
    }
  }, []);

  useEffect(() => {
    verifyReadiness();
  }, [verifyReadiness]);

  const handleManualRetry = async () => {
    setIsRetryingReadiness(true);
    const isReady = await verifyReadiness();
    setIsRetryingReadiness(false);
    if (!isReady) {
      throw new Error('Backend is still offline');
    }
  };

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
        setError(err.message);
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
    <div className="min-h-dvh bg-canvas-base flex flex-col font-sans text-gray-200 antialiased selection:bg-brass selection:text-canvas-base">
      <Header onNewSearch={handleNewSearch} backendStatus={backendStatus} />

      <main className="flex-1">
        {backendStatus === 'checking' ? (
          <div className="min-h-[calc(100dvh-8rem)] flex flex-col items-center justify-center px-5 py-10 text-center space-y-4">
            <div className="relative">
              <div className="w-14 h-14 rounded-xl bg-canvas-surfaceHigh border border-white/[0.08] flex items-center justify-center text-brass shadow-lg">
                <Server className="w-7 h-7 text-brass opacity-80" />
              </div>
              <div className="absolute inset-0 flex items-center justify-center">
                <Loader2 className="w-16 h-16 text-brass/30 animate-spin" />
              </div>
            </div>
            <div className="space-y-1">
              <p className="font-serif italic text-lg text-white">
                Connecting to LegalRAG Cluster
              </p>
              <p className="font-mono text-xs text-slateSteel">
                Verifying AWS EC2 container readiness (/ready probe)...
              </p>
            </div>
          </div>
        ) : backendStatus === 'offline' ? (
          <BackendOfflineView
            onRetry={handleManualRetry}
            isRetrying={isRetryingReadiness}
          />
        ) : view === 'ask' ? (
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
      <footer className="border-t border-white/[0.06] bg-canvas-surface/40 pt-6 pb-[calc(1.5rem+env(safe-area-inset-bottom,0px))] text-center text-xs font-mono text-slateSteel">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>LegalRAG · Autonomous Judicial Retrieval Engine</span>
          <span className="text-slateSteel/80">
            100,000 High Court Judgments · 538,079 Chunks · BGE + BM25 + Cross-Encoder
          </span>
        </div>
      </footer>
    </div>
  );
};

export default App;
