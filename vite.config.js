import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import path from 'path';

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  // HTML 文件、src/、public/ 都在 frontend/ 下
  root: 'frontend',
  build: {
    outDir: '../dist',
    rollupOptions: {
      // 多入口：/ 对应 main.js，/config 对应 config-main.js
      input: {
        main: path.resolve(import.meta.dirname, 'frontend/index.html'),
        config: path.resolve(import.meta.dirname, 'frontend/config.html'),
      },
    },
  },
});
