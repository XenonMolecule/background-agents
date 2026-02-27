# Precursor: Participant Setup Guide

Hey! If you're reading this, you already have the repo cloned and the main
Python environment working (the observation pipeline runs, tasks get proposed).
This guide covers the **last-mile setup** so that background agents can actually
*execute* those proposed tasks.

---

## Quick diagnosis

If you see tasks being proposed but agents never produce results, run:

```bash
ls ~/Library/Application\ Support/precursor/logs/
```

Open the most recent `.log` file. If the last lines look like this and then
nothing else follows:

```
INFO  Processing request of type ListToolsRequest
INFO  Processing request of type ListResourcesRequest
INFO  Processing request of type ListPromptsRequest
```

...then one of the MCP servers is hanging during startup. The fix is to either
install the missing dependency **or** disable that server in the config.

---

## Step 1 — Install Node.js (required for 3 servers)

Three MCP servers (`filesystem`, `websearch`, `slides`) use `npx` / `node`.

```bash
# Check if already installed
node --version    # need 18+
npx --version
```

If not installed:

```bash
# Option A — Homebrew (recommended on macOS)
brew install node

# Option B — nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
nvm install 20
nvm use 20
```

After installing, **open a new terminal** and verify `node --version` works.

---

## Step 2 — Set up the slides server

```bash
cd src/precursor/mcp_servers/slides
npm run setup
```

This installs the Node dependencies the slides MCP needs. If you don't need
slide generation, you can skip this and disable it (Step 6 below).

---

## Step 3 — Environment variables (`.env`)

Make sure you have a `.env` file at the repo root with these keys:

```bash
# Required for the LLM calls that power the agents
OPENAI_API_KEY=sk-...

# Required for the web-search server
BRAVE_API_KEY=BSA...

# Required for the coding agent to create GitHub PRs
GITHUB_TOKEN=ghp_...

# Required for calendar-aware context
CALENDAR_ICS=https://...your_outlook_ics_link...
```

If you don't have one of these yet, that's fine — just leave it blank and
disable the corresponding server (Step 6).

---

## Step 4 — Google Drive credentials

The `drive` server needs OAuth credentials so agents can read/write your Google
Docs.

1. Ask Michael for the `credentials.json` file (or create your own via
   [Google Cloud Console](https://console.cloud.google.com/apis/credentials)).
2. Place `credentials.json` at the **repo root** (next to `requirements.txt`).
3. Run this **once** interactively to authorize and create `token.pickle`:

```bash
python -c "from precursor.mcp_servers.drive.server import get_drive_service; get_drive_service()"
```

A browser window will open asking you to log in to Google. After authorizing,
a `token.pickle` file is created and future runs use it automatically.

If you don't want Drive integration right now, skip this and disable the
server (Step 6).

---

## Step 5 — Docker (for the coding agent)

The `coder` server uses OpenHands, which needs Docker to run its sandbox.

```bash
# Check if Docker is running
docker info
```

If Docker isn't installed, grab [Docker Desktop](https://www.docker.com/products/docker-desktop/).
Make sure to **start Docker Desktop** before running the agent.

If you don't need the coding agent, skip this and disable it (Step 6).

---

## Step 6 — Disable servers you haven't set up yet

Open `src/precursor/config/mcp_servers.yaml`. For any server you haven't
configured, set `enabled: false`:

```yaml
servers:
  - id: gum
    load: "python -m precursor.mcp_servers.gum.server"
    enabled: true        # this one always works

  - id: drive
    load: "python -m precursor.mcp_servers.drive.server"
    enabled: false       # <-- set to false if no credentials.json

  - id: filesystem
    load: "npx -y @modelcontextprotocol/server-filesystem ~/Documents ~/Desktop"
    enabled: true        # works if Node.js is installed

  - id: coder
    load: "python -m precursor.mcp_servers.coder.server"
    enabled: false       # <-- set to false if no Docker / GITHUB_TOKEN

  - id: websearch
    load: "npx -y @brave/brave-search-mcp-server --transport stdio"
    enabled: false       # <-- set to false if no BRAVE_API_KEY

  - id: slides
    load: "node src/precursor/mcp_servers/slides/src/index.js"
    enabled: false       # <-- set to false if npm setup not run

  - id: fetch
    load: "python -m mcp_server_fetch --ignore-robots-txt --user-agent=Precursor/1.0"
    enabled: true        # works if mcp-server-fetch is pip-installed
```

The agent will run with whatever servers you have available. You can always
enable more later as you set them up.

---

## Step 7 — Verify it works

Run the agent in no-deploy mode first to confirm MCP servers load:

```bash
python -m precursor.main --mode gum --no-deploy --max-steps 2 --log-level INFO
```

In the output, look for:
- `mcp_loader: server 'X' started successfully` (good)
- `mcp_loader: server 'Y' failed to start — skipping` (that server needs setup)

Once you're happy with which servers load, remove `--no-deploy` to let agents
execute tasks.

---

## Step 8 — Check agent logs

Agent logs are written to:

```
~/Library/Application Support/precursor/logs/
```

Each deployed agent gets its own timestamped log file. If an agent fails, the
log will contain the traceback. If a server times out (default: 60s), you'll
see:

```
MCP server 'X' did not initialize within 60s
```

You can increase the timeout in `src/precursor/config/settings.yaml`:

```yaml
server_startup_timeout: 120  # seconds
```

---

## Checklist

| Step | What | How to verify |
|------|------|---------------|
| Python env | `pip install -r requirements.txt && pip install -e .` | `python -c "import precursor"` |
| Node.js 18+ | `brew install node` | `node --version` |
| `.env` keys | `OPENAI_API_KEY`, etc. | `cat .env` |
| Google Drive | `credentials.json` + `token.pickle` at repo root | Files exist |
| Docker | Docker Desktop running | `docker info` |
| Slides | `npm run setup` in slides dir | `ls src/precursor/mcp_servers/slides/node_modules` |
| Disable unused | `enabled: false` in `mcp_servers.yaml` | No more hangs |
