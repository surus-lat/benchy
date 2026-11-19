# benchy website

Source of https://benchy.lat — Vite + React + Tailwind.

## Develop

```bash
npm ci
npm run dev
```

## Build

```bash
npm run build   # outputs to dist/
```

## Keeping the site honest about the engine

Every code example shown on the site is validated against the real engine
by CI (`.github/workflows/website-checks.yml`) on any change to `benchy/`
or `website/`:

```bash
pip install -e ..                          # the engine, from the repo root
python checks/validate_site_examples.py . ..
```

It extracts every code block from `src/pages/Index.tsx` and:
- compiles the full `benchmark.yaml` with `benchy compile`,
- splices each partial YAML block (program / scoring / ai-system) into the
  canonical example and compiles again,
- checks every `benchy <subcommand>` exists in the CLI,
- parses the JSON examples.

If the engine changes and the site would start lying, the check fails.
When you change the engine, run this and update `src/pages/Index.tsx`
accordingly.

## Deploy

Push to `main` touching `website/**` deploys to Cloudflare Pages via
`.github/workflows/website-deploy.yml` (project `benchy`).
Requires repo secrets `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`.
