## Project Scratchpad Generator

Sequentially updates and logs a project scratchpad per timestamp by joining the context log with project predictions. Each output row contains the full scratchpad snapshot and the agent's edit summary.

### Requirements
- Python 3.10+
- `pip install -r requirements.txt`
- Environment variables: `OPENAI_API_KEY` (or `DSPY_API_KEY`) and optionally `DSPY_MODEL`

### Default Inputs
- Context log: `dev/survey/context_log.csv`
- Predictions CSV (default): `dev/survey/analysis/project_classification/results/context_project_predictions_openai_gpt-5_with_history.csv`

### Output
- Writes a CSV to `dev/survey/analysis/project_scratchpad/results/project_scratchpad_<model>.csv`
- Also writes/overwrites `dev/survey/analysis/project_scratchpad/results/project_scratchpad_latest.csv`

Columns: `timestamp,project,scratchpad,summary,model`

### Usage

Basic run (uses defaults and with_history predictions):

```bash
python dev/survey/analysis/project_scratchpad/project_scratchpad.py \
  --model openai/gpt-5
```

Limit to the first N rows for quick testing:

```bash
python dev/survey/analysis/project_scratchpad/project_scratchpad.py \
  --model openai/gpt-5 \
  --limit 50
```

Specify inputs and tune objective aggregation:

```bash
python dev/survey/analysis/project_scratchpad/project_scratchpad.py \
  --context-log dev/survey/context_log.csv \
  --predicted-project-csv dev/survey/analysis/project_classification/results/context_project_predictions_openai_gpt-5_with_history.csv \
  --model openai/gpt-5 \
  --current-goals-top-k 3 \
  --former-goals-window 10 \
  --former-top-per-row 1 \
  --former-max-total 10
```

### Slides Integration

Open `dev/survey/slides/index.html` in a browser. The app will load:
- Context: `../context_log.csv`
- Predictions: `../analysis/project_classification/results/context_project_predictions_latest.csv` (configurable via `?pred=...`)
- Scratchpad: `../analysis/project_scratchpad/results/project_scratchpad_latest.csv` (configurable via `?spad=...`)

Scratchpad content and edit summaries are shown as collapsible sections on context slides.


