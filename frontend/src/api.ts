import { QueryRequest, QueryResponse } from './types';
import { getActiveBaseUrl, API_TIMEOUT_MS } from './config';

export class ApiError extends Error {
  status?: number;
  code?: string;
  constructor(message: string, status?: number, code?: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

/**
 * Executes a query against the active LegalRAG backend.
 * AbortSignal supports user-triggered cancellation.
 */
export async function queryLegalRag(
  req: QueryRequest,
  signal?: AbortSignal
): Promise<QueryResponse> {
  const baseUrl = getActiveBaseUrl();
  const url = `${baseUrl.replace(/\/+$/, '')}/query`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT_MS);

  // Link external abort signal if provided
  if (signal) {
    signal.addEventListener('abort', () => controller.abort());
  }

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question: req.question }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      if (res.status === 422) {
        throw new ApiError('Question must be between 3 and 4,000 characters.', 422);
      } else if (res.status === 500) {
        throw new ApiError('Document retrieval failed on server.', 500);
      } else if (res.status === 502) {
        throw new ApiError('Answer generation failed on server provider.', 502);
      } else if (res.status === 503) {
        throw new ApiError('Backend service is warming up. Please retry shortly.', 503);
      }
      throw new ApiError(`Request failed with status ${res.status}`, res.status);
    }

    const data: QueryResponse = await res.json();
    return data;
  } catch (err: unknown) {
    clearTimeout(timeoutId);
    if (err instanceof ApiError) throw err;
    if (err instanceof Error && err.name === 'AbortError') {
      throw new ApiError('Request timed out or was cancelled.', 408, 'ABORT_ERR');
    }
    throw new ApiError(
      err instanceof Error ? err.message : 'Unable to connect to backend server.'
    );
  }
}
