# background-agents
Repo for the Background Agents Rotation Project

### Setup

- Python 3.10+
- Recommended (venv) and editable install:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Note: `pip install -e .` installs the `precursor` package in editable mode so imports/CLIs work during development.

### Configure environment

1) Copy `.env.example` to `.env` and fill these keys:
   - `OPENAI_API_KEY` — OpenAI API key
   - `ANTHROPIC_API_KEY` - Anthropic API key
   - `BRAVE_API_KEY` - Brave Search API key (for adding web search to agents)
   - `CALENDAR_ICS` — HTTPS ICS URL to your calendar feed
   - `GITHUB_TOKEN` — GitHub Personal Access Token (repo scope)

### Additional prerequisites

1. Node.js 18+ (for MCP servers and slides)
   - Install via nvm or package manager; ensure `node` and `npx` are available.
2. Docker (recommended for the coding agent)
   - Required by OpenHands to run the sandbox container for code edits and PRs.
3. Xcode 15+ (or Command Line Tools) on macOS
   - Required to build and run the Swift app (`PrecursorApp`).

### 🔗 Get an ICS Link for Outlook

1. Open Outlook on the web:  
   [outlook.office.com](https://outlook.office.com)

2. Go to **Settings (⚙️)** → **View all Outlook settings** → **Calendar** → **Shared calendars**.

3. Under **Publish a calendar**, choose your calendar → set permission (**Can view all details**) → click **Publish**.  
   Links: [Share your calendar](https://support.microsoft.com/en-us/office/share-your-calendar-in-outlook-on-the-web-7ecef8ae-139c-40d9-bae2-a23977ee58d5) · [Publishing overview](https://support.microsoft.com/en-us/office/introduction-to-publishing-internet-calendars-a25e68d6-695a-41c6-a701-103d44ba151d)

4. Copy the **ICS** link (should end with `.ics`). If it begins with `webcal://`, change it to `https://`.

5. Set this in your `.env` as:  
   `CALENDAR_ICS="https://…your_link.ics"`

### 🔐 Get a GitHub token

1. Go to GitHub → Settings → Developer settings → Personal access tokens.  
2. Create a Fine-grained (preferred) token with access to the repos you’ll use.  
3. Set repository permissions:
   - Contents: Read and write
   - Pull requests: Read and write
   - Workflows: Read and write
   - Commit statuses: Read
   - Metadata: Read
   - (Optional) Issues: Read
4. Copy the token and add to `.env` as `GITHUB_TOKEN`.  
5. Links: [New fine-grained token](https://github.com/settings/personal-access-tokens/new) · [Classic tokens](https://github.com/settings/tokens)

### 🗂 Google Drive credentials (for Drive/Docs tools)

1. Open the Google Cloud Console.  
2. Create/select a project and enable the “Google Drive API” and the "Google Docs API"
3. Create OAuth client credentials (Desktop app), download the JSON as `credentials.json`.  
4. Place `credentials.json` at the project root (or set `GOOGLE_CREDENTIALS_JSON`).  
5. On first use, you’ll be prompted to authorize; a `token.pickle` will be created (or set `GOOGLE_TOKEN_PICKLE`).  

Links: [Enable Drive API](https://console.cloud.google.com/apis/library/drive.googleapis.com) · [Credentials](https://console.cloud.google.com/apis/credentials)

Scopes required (add on the OAuth consent screen):
- `https://www.googleapis.com/auth/drive`
- `https://www.googleapis.com/auth/documents`

Defaults for Google files are set in `src/precursor/config/mcp_servers.yaml`, so placing both files at the repo root works out of the box.

### Enable web search (Brave)

1) Create a Brave Search API key and add to `.env`:
```bash
BRAVE_API_KEY=your_key_here
```
2) No local install needed; the project uses the Brave Search MCP via `npx`.

Docs: [Brave Search API overview](https://search.brave.com/help/api) · [Manage API keys](https://api.search.brave.com/app/keys)

### Enable the agent to create slides

```bash
cd src/precursor/mcp_servers/slides
npm run setup
```

### Build/Install the user interface (notifications + displaying work)

1) Install notifier:
```bash
brew install terminal-notifier
```

2) Build the app (once):
```bash
cd src/interface/PrecursorApp
swift build
```

### Config files (what to edit)

1. `src/precursor/config/projects.yaml`
   - Add your projects under `projects:` with a unique `name` and short `description`.
   - Toggle `agent_enabled` per project to allow background actions.
   - The CLI and scratchpad use the `name` exactly, so keep it stable.
2. `src/precursor/config/user.yaml`
   - Set `name`, update `description`, and refine `agent_goals`.  
   - `name` is also injected as `USER_NAME` for the GUM MCP.
3. `src/precursor/config/settings.yaml`
   - Tune decision-making: `value_weight`, `feasibility_weight`, `user_preference_alignment_weight`.
   - Control deployment: `max_deployed_tasks`, `deployment_threshold`, `safety_threshold`.
   - Raise thresholds to be more conservative; lower to be more active.
4. `src/precursor/config/mcp_servers.yaml`
   - Enable/disable servers under `servers:`; no Python changes needed.

All besides the MCP servers can actually be edited from the user interface.

### Advanced configuration (optional)

- To override built-in config files, set:
  - `PRECURSOR_PROJECTS_FILE`, `PRECURSOR_SETTINGS_FILE`, `PRECURSOR_MCP_SERVERS_FILE`
- To change the scratchpad DB location:
  - `PRECURSOR_SCRATCHPAD_DB=/path/to/scratchpad.db`
- To customize slide export location (used by the slides MCP):
  - `SLIDEV_DIR=/path/for/generated/slides`

User profile is read from `src/precursor/config/user.yaml` (used to inject `USER_NAME` for the GUM server).

### Run the agent (GUM mode)

GUM mode streams your real-time context into the agent. Ensure `.env` is set (keys above), then:

```bash
python -m precursor.main --mode gum --log-level INFO
```

Optional logging:
- Write processed events + final scratchpad to CSV:
```bash
python -m precursor.main --mode gum --output-csv dev/survey/pipeline_run.csv
```
- Also log candidate tasks scored by the agent:
```bash
python -m precursor.main --mode gum --agent-output-csv dev/survey/pipeline_run.agent_candidates.csv
```
- Save per-event screenshots (GUM mode) and include their paths in the CSV (`screenshot_path`):
```bash
python -m precursor.main --mode gum --screenshot-dir dev/survey/screenshots --output-csv dev/survey/pipeline_run.csv
```
- All logs + screenshots:
```bash
python -m precursor.main --mode gum --output-csv dev/survey/pipeline_run.csv --agent-output-csv dev/survey/pipeline_run.agent_candidates.csv --screenshot-dir dev/survey/screenshots
```

Other useful flags:
- `--no-deploy` — score/log tasks without executing next steps
- `--max-steps N` — stop after N events
- `--force-reset` — delete the scratchpad DB on startup

### Testing

#### Running Integration Tests

The project includes deterministic integration tests for project transitions and notifications:

```bash
# Run project transition integration tests
pytest -k project_transition
```

Or run the specific integration test file:

```bash
pytest tests/test_project_transition_integration.py
```

The integration tests use a deterministic CSV simulator mode to create hermetic test scenarios without sleeps or network calls. They test:

1. **Project transition detection** - Verifies transitions between projects trigger appropriate managers
2. **Batch processing metrics** - Ensures telemetry events are emitted for observation processing  
3. **Notification skipping** - Tests that notifications are skipped when no pending agent tasks exist
4. **Notification sending** - Verifies notifications are sent when pending agent tasks are present

#### Test Architecture

The tests use mock sinks (`MockTelemetrySink` and `MockNotificationSink`) to capture telemetry events and notifications for assertions, avoiding brittle log string matching. The CSV simulator runs in `step` mode for fully synchronous, deterministic execution.
