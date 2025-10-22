# Slides: Context + Survey

This viewer generates slides from two sources and orders them chronologically:
- `dev/survey/context_log.csv` (rich observer snapshots)
- `dev/survey/survey_responses.csv` (self-reports)

It renders two slide types with collapsible sections for long text and supports PNG/PPTX export.

## Quick start

1) Start a static server from the REPO ROOT (required so paths resolve):

```bash
# from /Users/michaelryan/Documents/School/Stanford/Research/background-agents
python3 -m http.server 8000
# or
npx serve -l 8000
```

2) Open the slides viewer:

```text
http://localhost:8000/dev/survey/slides/
```

- The app loads `../context_log.csv` and `../survey_responses.csv` relative to `slides/`, which resolves to `/dev/survey/context_log.csv` and `/dev/survey/survey_responses.csv` when served from the repo root.
- Screenshot URLs are resolved under `/dev/survey/screenshots/...`, which also assumes repo root hosting.
 - Optional: if present, `dev/survey/context_project_predictions.csv` will be loaded and joined by `timestamp` to display a Predicted Project on context slides.

### Useful URL parameters

- `?scale=3` — set export scale (1–5).
- `?i=10` — jump to slide index 10 (0‑based).
- `?ts=2025-10-17` — jump to first slide whose timestamp display contains the substring.
- `?refresh=1` (or `?nocache=1` / `?force=1`) — cache‑bust CSV and screenshot fetches.

You can also click the "Refresh" button in the toolbar to force a reload with cache‑busting.

## Using the viewer

- Prev/Next: navigate between slides (context and survey slides are interleaved by timestamp).
- Exports:
  - Export PNG (current): downloads the visible slide.
  - Export PNG (all): iterates and downloads all slides.
  - Export PPTX (all): builds a 16:9 deck from all slides.
- Optional scale parameter for crisper exports (1–5):

```text
http://localhost:8000/dev/survey/slides/?scale=3
```

## What’s shown

- Context slides: screenshot (left), right panel with Inferred Goals (compact cards) and collapsible sections for Propositions, Calendar, Context Update, Recent Observations, and Reasoning.
- Survey slides: project/task and the four “helpful” fields, grouped for quick reading.

## Notes / Troubleshooting

- If screenshots don’t load, ensure you’re serving from the repo root (not `dev/survey/`).
- If your local repo path differs, update `REPO_ROOT_PREFIX` in `slides/app.js` so absolute screenshot paths are correctly relativized.
- Very large CSVs can be heavy to parse client-side; consider filtering before loading if needed.


