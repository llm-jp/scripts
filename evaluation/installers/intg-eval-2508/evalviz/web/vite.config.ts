import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import { tanstackRouter } from '@tanstack/router-plugin/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// 開発時は evalviz サーバ (python -m evalviz_server) の /api に中継する。EVALVIZ_API で向き先を変えられる
export default defineConfig({
  plugins: [
    // ルーター plugin は react plugin より前に置く (src/routes/ からルートツリー src/routeTree.gen.ts を生成する)
    tanstackRouter({ target: 'react', autoCodeSplitting: true }),
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: { '@': path.resolve(import.meta.dirname, './src') },
  },
  server: {
    proxy: { '/api': process.env.EVALVIZ_API ?? 'http://127.0.0.1:8791' },
  },
})
