/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_API_VERSION: string;
  readonly VITE_WS_URL: string;
  readonly VITE_APP_NAME: string;
  readonly VITE_APP_VERSION: string;
  readonly VITE_ENABLE_REAL_TIME_UPDATES: string;
  readonly VITE_ENABLE_ANALYTICS: string;
  readonly VITE_ENABLE_DEBUG: string;
  readonly VITE_ENABLE_MOCK_AUTH: string;
  readonly VITE_TOKEN_REFRESH_INTERVAL: string;
  readonly VITE_DEFAULT_LEVERAGE: string;
  readonly VITE_MAX_LEVERAGE: string;
  readonly VITE_MIN_TRADE_SIZE: string;
  readonly VITE_MAX_TRADE_SIZE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
