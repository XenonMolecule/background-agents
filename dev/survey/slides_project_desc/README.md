# Project Description Timeline Viewer

Lightweight static web app to browse inferred project descriptions over time, joined with the original context log and pipeline screenshot paths. Click through each step to see the screenshot and the evolving description.

## Files
- `index.html` – the app entry
- `app.js` – loads and joins CSVs, renders UI
- `style.css` – simple styles

## Default inputs
- Descriptions: `../data_collection/12_3_experiments/project_descriptions.csv`
- Log: `../data_collection/11_20_experiments/no_user/log.csv`
- Pipeline: `../pipeline_run.csv`

These can be overridden via query params:

```
?desc=REL_OR_ABS_PATH_TO_DESCRIPTIONS_CSV&log=...&pr=...&project=MyProject
```

Optionally set a `root` param to strip absolute screenshot paths to repo-relative paths (useful if your images are addressed with absolute paths):

```
?root=/Users/michaelryan/Documents/School/Stanford/Research/background-agents
```

## Running locally
Serve this folder (or the repo root) with any static server, for example:

```bash
cd dev/survey/slides_project_desc
python3 -m http.server 8080
# then open http://localhost:8080 in your browser
```

If serving from repo root, open:

```
http://localhost:8080/dev/survey/slides_project_desc/index.html
```

You can preselect a project and force-refresh cached CSVs:

```
http://localhost:8080/dev/survey/slides_project_desc/index.html?project=AutoMetrics%20Release&refresh=1
```

## UI notes
- Top-left: choose Project; Step lets you jump to a specific row
- Prev/Next navigates the sequence; counter shows position
- Right panel prioritizes “Inferred Project Description” for the current step; extras are collapsible






