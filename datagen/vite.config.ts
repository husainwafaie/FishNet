import { defineConfig, type Plugin } from 'vite';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DATASET_ROOT = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  '..',
  'dataset',
);

/**
 * Dev-server endpoint the browser posts captured samples to. Writes
 * <name>.png + <name>.json pairs under dataset/images and dataset/labels,
 * sidestepping the browser's one-download-per-file friction for large batches.
 */
function datasetWriter(): Plugin {
  return {
    name: 'dataset-writer',
    configureServer(server) {
      server.middlewares.use('/api/save', (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end();
          return;
        }
        let body = '';
        req.on('data', (chunk) => (body += chunk));
        req.on('end', () => {
          try {
            const { name, imageBase64, label } = JSON.parse(body) as {
              name: string;
              imageBase64: string;
              label: unknown;
            };
            if (!/^[\w-]+$/.test(name)) throw new Error(`bad sample name: ${name}`);
            fs.mkdirSync(path.join(DATASET_ROOT, 'images'), { recursive: true });
            fs.mkdirSync(path.join(DATASET_ROOT, 'labels'), { recursive: true });
            fs.writeFileSync(
              path.join(DATASET_ROOT, 'images', `${name}.png`),
              Buffer.from(imageBase64, 'base64'),
            );
            fs.writeFileSync(
              path.join(DATASET_ROOT, 'labels', `${name}.json`),
              JSON.stringify(label, null, 2),
            );
            res.statusCode = 200;
            res.end('ok');
          } catch (err) {
            res.statusCode = 400;
            res.end(String(err));
          }
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [datasetWriter()],
  server: {
    port: 5175,
    // Windows 8.3 short paths (e.g. HUSAIN~1) don't match Vite's resolved
    // allow-list entries; this is a local-only dev tool, so relax the check
    fs: { strict: false },
  },
});
