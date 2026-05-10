import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Load .env so VITE_API_PROXY_TARGET is available at config time.
  // This is needed because vite.config.js runs before the Vite env injection.
  const env = loadEnv(mode, process.cwd(), '')

  // In Docker the backend is reachable at http://backend:8000.
  // Locally it's http://127.0.0.1:8000.
  // Override with VITE_API_PROXY_TARGET in .env or docker-compose environment.
  const apiTarget = env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000'

  return {
    plugins: [react()],
    server: {
      host: '0.0.0.0',   // required when running inside Docker
      port: 5173,
      proxy: {
        // All /api/* requests are forwarded to the FastAPI backend.
        // The backend has no /api prefix, so we strip it.
        '/api': {
          target: apiTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      },
    },
  }
})
