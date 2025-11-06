import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { ServerResponse } from 'http';

// https://vitejs.dev/config/
export default defineConfig({
  base: '/INF3/',
  plugins: [react()],
  server: {
    watch: {
      // Ignore changes to tsconfig.json to prevent unwanted server restarts
      // if an IDE or another tool modifies it.
      ignored: ['**/tsconfig.json'],
    },
    proxy: {
      // Proxy for the local backend, which handles both knowledge base
      // and AI generation requests during development.
      '/api': {
        target: 'http://localhost:5000', // Use localhost for better compatibility.
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        // Add logging to help debug proxy issues.
        configure: (proxy, options) => {
          proxy.on('proxyReq', (proxyReq, req, res) => {
            console.log(`[vite-proxy] Sending request: ${req.method} ${req.url} -> ${options.target}${proxyReq.path}`);
          });
          proxy.on('proxyRes', (proxyRes, req, res) => {
            console.log(`[vite-proxy] Received response: ${proxyRes.statusCode} ${req.url}`);
          });
          proxy.on('error', (err, req, res) => {
            console.error('[vite-proxy] Error:', err);
            // FIX: Use the imported ServerResponse for a proper type guard to resolve TypeScript errors on `res` which could be a Socket.
            if (res instanceof ServerResponse && !res.headersSent) {
              res.writeHead(502, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ message: 'Proxy Error', error: err.message }));
            }
          });
        }
      },
    }
  },
})