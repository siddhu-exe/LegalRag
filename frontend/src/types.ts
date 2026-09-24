/**
 * Strict TypeScript types for LegalRAG Frontend integration.
 * Invariant contract defined in docs/FRONTEND_INTEGRATION.md
 */

export interface QueryRequest {
  question: string;
}

export interface Citation {
  chunk_id: string; // Guaranteed non-null unique key
  cnr: string | null;
  court_code: string | null;
  decision_date: string | null;
  title: string | null;
}

export type QueryStatus = 'ok' | 'generation_error' | 'retrieval_error';

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  retrieved_chunk_ids: string[];
  retrieve_ms: number;
  rerank_ms: number;
  generate_ms: number;
  total_ms: number;
  status: QueryStatus;
}

export type BackendTarget = 'local' | 'deployed';

export type ActiveView = 'ask' | 'results';
