import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
  build: {
    // Default: outputs to web/frontend/dist/
    // Docker copies this to /app/web/backend/static/ in the runtime stage.
    // For local dev builds: npm run build && cp -r dist ../backend/static
    outDir: 'dist',
    emptyOutDir: true,
  },
})
