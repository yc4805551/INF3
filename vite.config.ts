import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  // Set the base path for deployment to the specific GitHub repository.
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
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    }
  },
})
