# Project Classification (Predictions CSV)

Batch classify each `context_log.csv` row into one of the predefined project labels and write a CSV you can join by `timestamp`.

## Labels

- Personalization Dataset Collection
- AutoMetrics Release
- NLP Retreat Planning
- Background Agents
- Logistics
- Health
- HW3 Assignment Creation
- Misc

## Usage

From repo root:

```bash
# Pass model explicitly; default output filename will include the model name
python -m dev.survey.analysis.project_classification.project_classifier --input dev/survey/context_log.csv --model openai/gpt-5 --limit 10
```

- Output schema: `timestamp,predicted_project,model` (minimal, join by timestamp)
- Join key: `timestamp` (matches `context_log.csv` column)
- The slides viewer will automatically pick up `dev/survey/context_project_predictions.csv` if present (or pass `?pred=path` to the slides URL) and show "Predicted Project" on context slides.

### Using recent history (last 10 minutes, top-3)

```bash
python -m dev.survey.analysis.project_classification.project_classifier --input dev/survey/context_log.csv --model openai/gpt-5 --use-history --history-window-minutes 10 --history-top-k 3
```

- This writes a suffixed file: `...context_project_predictions_<model>_with_history.csv`, and also updates `.../results/context_project_predictions_latest.csv`.

## Notes
- The runner extracts inputs from `context_log.csv` fields: `goals`, `user_details`/`recent_observations`, `calendar_events`, and `screenshot_path`.
- If a prediction fails, it falls back to `Misc`.
- You can tweak temperature with `--temperature`.

## Timeline plotting

Produce a two-row timeline image: human self-reports (top) and optional AI predictions (bottom). The human timeline assumes each survey reading continues for a horizon (default 45 min) unless interrupted by a later reading; gaps remain blank. The AI timeline uses its own shorter horizon (default 4 min). Day breaks are marked at local midnights.

Install deps (from repo root):

```bash
pip install -r requirements.txt
```

Human-only (uses `dev/survey/survey_responses.csv` by default):

```bash
python -m dev.survey.analysis.project_classification.timeline_plot --output dev/survey/analysis/project_classification/timeline.png
```

With AI predictions (CSV schema: `timestamp,predicted_project,model`):

Defaults already match your files, so you can keep it simple:

```bash
python -m dev.survey.analysis.project_classification.timeline_plot --ai-csv dev/survey/analysis/project_classification/results/context_project_predictions_latest.csv --output dev/survey/analysis/project_classification/results/timeline_with_ai.png
```

Compressed:
```bash
python -m dev.survey.analysis.project_classification.timeline_plot --ai-csv dev/survey/analysis/project_classification/results/context_project_predictions_latest.csv --output dev/survey/analysis/project_classification/timeline_with_ai.png --compact-output dev/survey/analysis/project_classification/results/timeline_with_ai_compact.png --compact-gap 60
```

With-history comparison (adds a third row and shows per-row accuracies inline):

```bash
python -m dev.survey.analysis.project_classification.timeline_plot --ai-csv dev/survey/analysis/project_classification/results/context_project_predictions_openai_gpt-5.csv --ai-history-csv dev/survey/analysis/project_classification/results/context_project_predictions_openai_gpt-5_with_history.csv --output dev/survey/analysis/project_classification/results/timeline_with_ai.png --compact-output dev/survey/analysis/project_classification/results/timeline_with_ai_compact.png
```

Notes on accuracy:
- Accuracy is exact-match of AI label at each AI timestamp against the human label active at that time.
- Each AI timeline title includes its own accuracy (e.g., `AI-predicted timeline — acc 72.3% (n=145)`).

Fully explicit (defaults shown):

```bash
python -m dev.survey.analysis.project_classification.timeline_plot --ai-csv dev/survey/analysis/project_classification/results/context_project_predictions_latest.csv --ai-col predicted_project --ai-ts-format %Y%m%d_%H%M%S --ai-naive-origin local --ai-horizon 4 --output dev/survey/analysis/project_classification/results/timeline_with_ai.png
```

Common options:

- `--input` Path to human survey CSV (default: `dev/survey/survey_responses.csv`)
- `--output` Output image path (PNG)
- `--tz` Display timezone (default: `US/Pacific`)
- `--horizon` Human carryover minutes (default: 45)
- `--width`/`--height` Figure size in pixels (default: 1400x600)
- `--ai-csv` Path to AI predictions CSV (enables second row)
- `--ai-col` Column name for AI project (default: `predicted_project`)
- `--ai-horizon` AI carryover minutes (default: 4)
- `--ai-ts-format` Timestamp format for AI CSV (default: `%Y%m%d_%H%M%S`)
- `--ai-naive-origin` Origin for naive AI timestamps when using `--ai-ts-format` (`local`|`utc`, default: `local`)

Label normalization:
- Both human and AI project labels are canonically mapped into the TRUE_PROJECTS set shown above (e.g., "Autometrics"/"AutoMetrics" → "AutoMetrics Release"; "NLP Retreat" → "NLP Retreat Planning"; "Personalization"/"Longitudinal Personalization" → "Personalization Dataset Collection"; "HW3 Prep" → "HW3 Assignment Creation"; "In between efforts (Misc)" → "Misc").
- Unknown labels fall back to their trimmed original and still render.
