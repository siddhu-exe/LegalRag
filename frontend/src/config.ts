export const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  import.meta.env.VITE_API_BASE_URL ||
  '';

export const API_TIMEOUT_MS =
  Number(import.meta.env.VITE_API_TIMEOUT_MS) || 120000;

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}
