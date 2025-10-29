import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  root: 'collectra/editor/static/src',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    rollupOptions: {
      input: 'collectra/editor/static/src/main.jsx',
      output: {
        entryFileNames: 'editor.js',
        assetFileNames: 'editor.css'
      }
    }
  }
});