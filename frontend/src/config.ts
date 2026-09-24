import { BackendTarget } from './types';

export const LOCAL_URL = import.meta.env.VITE_API_LOCAL || 'http://localhost:7860';
export const DEPLOYED_URL = import.meta.env.VITE_API_DEPLOYED || '';
export const DEFAULT_BACKEND: BackendTarget =
  (import.meta.env.VITE_DEFAULT_BACKEND as BackendTarget) || 'local';
export const API_TIMEOUT_MS =
  Number(import.meta.env.VITE_API_TIMEOUT_MS) || 120000;

const STORAGE_KEY = 'legalrag_backend_target';

export function getActiveBackend(): BackendTarget {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === 'local' || stored === 'deployed') {
    return stored;
  }
  return DEFAULT_BACKEND;
}

export function setActiveBackend(target: BackendTarget): void {
  localStorage.setItem(STORAGE_KEY, target);
}

export function getActiveBaseUrl(): string {
  const target = getActiveBackend();
  if (target === 'deployed') {
    return DEPLOYED_URL || LOCAL_URL;
  }
  return LOCAL_URL;
}

export function isDeployedConfigured(): boolean {
  return Boolean(
    DEPLOYED_URL &&
    DEPLOYED_URL !== 'https://your-deployed-backend-url.com' &&
    !DEPLOYED_URL.includes('your-deployed-backend-url')
  );
}
