/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_LOCAL?: string;
  readonly VITE_API_DEPLOYED?: string;
  readonly VITE_DEFAULT_BACKEND?: string;
  readonly VITE_API_TIMEOUT_MS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
