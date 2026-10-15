# Unstoppable AI Workforce v3

Claude = **manager**. Gemini Flash / Gemini Pro / Grok (or Groq) = **workers**.
v3 works on **your existing local projects** without uploading them and without GitHub.
Claude never reads your code: the server on your PC reads the folder, picks the relevant files,
and gives them to the workers. Claude only sees short summaries, so it uses far fewer tokens.

---------------------------------------------------------------------------

## 1. One-time setup (Windows, about 5 minutes)

1. Install Python 3.10+ from https://www.python.org/downloads/ (tick **Add python.exe to PATH**).
2. Unzip this folder somewhere permanent, e.g. `C:\ai_workforce`.
3. Get your keys (free tiers are fine):
   - Gemini: https://aistudio.google.com/apikey  (make a 2nd key with another Google account if you like)
   - Groq: https://console.groq.com (key starts with `gsk_`)  or xAI Grok: https://console.x.ai (`xai-`)
4. **Close Claude Desktop completely** (system tray -> Quit).
5. Double-click **`setup_and_run.bat`**. It installs everything, then asks you:
   - Gemini key A / key B / Groq-or-xAI key. Typing is hidden: right-click to paste, press Enter.
     Press Enter alone to skip a key.
   - Project folders: paste a folder that CONTAINS your projects (e.g. `C:\Users\you\Projects`),
     or a single project folder. Empty line = done.
   - Allow running tests? Say `y` if you want `run_check` (tests/build + auto-fix). Say `n` to keep it off.
   - Add to Claude Desktop automatically? `y`.
   - Test connections? `y` (it makes a tiny test project, edits it, undoes it).
6. Start Claude Desktop. Settings -> Developer: `ai_workforce` should say **running**.
7. Claude Desktop -> Settings -> "Instructions for Claude" (profile preferences): paste the text of
   **`PASTE_INTO_CLAUDE_SETTINGS.txt`**. Done. You never repeat it in new chats.

Your keys live only in `aiw_config.json` next to `server.py` (never in the Claude config, never in chat).
If you ever pasted a key anywhere public, revoke it and make a new one.

## 2. Using it

In a new chat, start with: **"Act as manager."** Claude asks: *New project or existing folder?*

**A) Fresh task (new project)**
> Act as manager. New project "shop-api": FastAPI + SQLite, products and orders endpoints, with tests.

Claude calls `open_project(new="shop-api", ...)` and the workers build it in `ai_workforce\workspace\shop-api`.

**B) Existing project on your PC**
> Act as manager. Existing project: C:\Users\me\Projects\pair-match. Review the backend for security bugs and fix the serious ones.

Claude calls `open_project(project="C:\\Users\\me\\Projects\\pair-match")`, then `review()`, then `work(task="fix F1,F2")`.
If the folder is not allowed yet, double-click **`add_project.bat`** (or drag the folder onto it). No restart needed.

Things you can ask in mode B: explain how login works, find bugs, add a feature, change the backend, refactor,
write tests, fix failing tests, update many files at once, "preview the change first", "undo that".

## 3. Tools (Claude uses these; you just talk)

| Tool | What it does |
|---|---|
| `open_project` | open a local folder (must be allowed) or create a new project; stores a short brief for the workers |
| `ask` | worker reads the relevant files and answers (read-only) |
| `review` | audits the whole project in parallel batches, verifies findings, returns F1..Fn |
| `work` | edit or create code on disk (search/replace edits for existing files); `review=True`, `apply=False` supported |
| `swarm` | up to 3 independent `work` tasks in parallel (different files) |
| `run_check` | runs your tests/build (only if allowed) and auto-fixes failures |
| `changes` | list / diff / apply (a proposal) / undo |
| `view`, `patch` | tiny debugging helpers |
| `status` | which model handled what, timings, fallbacks |

## 4. Safety

- Workers can only touch folders you allowed. `..`, absolute paths and symlinks out of the project are refused.
- `.env`, `*.pem`, `id_rsa`, `secrets.*`, service-account files etc. are **never read**; secret-looking values
  (API keys, tokens, passwords in code) are replaced by `<<REDACTED_SECRET>>` before anything goes to a model.
- `node_modules`, `venv`, `.git`, `build`, `dist`, binaries and `.gitignore`d files are skipped.
- Every change is backed up in `.aiw_data\<project>\changes\cNNN\`. `changes(action="undo")` restores it, and refuses to
  overwrite your own later edits unless forced.
- Quality gates before anything is written: Python/JSON syntax, no "...rest of code" placeholders, no truncated rewrites,
  only the allowed files. A bad answer is retried with the error; if it still fails, **nothing is changed**.
- Commands run only through `run_check`, only if you enabled it, with your key variables removed from the environment.
- Your project folder is never modified by the server except through these checked changes.

Note: file contents that are sent to the workers go to Google / Groq / xAI. Do not allow folders that must stay private.

## 5. Settings file (`aiw_config.json`) and manual Claude config

The wizard writes both. Manual alternative: put this into
`%APPDATA%\Claude\claude_desktop_config.json` (keep the other entries, mind the commas):

```json
{
  "mcpServers": {
    "ai_workforce": {
      "command": "C:\\ai_workforce\\venv\\Scripts\\python.exe",
      "args": ["C:\\ai_workforce\\server.py"],
      "env": {
        "GEMINI_API_KEY_A": "PASTE_GEMINI_KEY_A_HERE",
        "GEMINI_API_KEY_B": "PASTE_GEMINI_KEY_B_HERE",
        "XAI_API_KEY": "PASTE_GROQ_OR_XAI_KEY_HERE",
        "XAI_BASE_URL": "https://api.groq.com/openai/v1",
        "AIW_PROJECTS": "C:\\Users\\you\\Projects"
      }
    }
  }
}
```

Use `https://api.x.ai/v1` for xAI. Remove the `env` block if you use the wizard (keys then live in `aiw_config.json`).

Useful optional settings (in `aiw_config.json` or as env variables with the `AIW_` prefix):
`allow_run`, `fix_rounds`, `max_fixes`, `flash_models`, `pro_models`, `grok_model`, `gemini_backend`
(`auto`/`developer`/`vertex`), `review_max_batches`, `keep_changes`.

Commands (from a Command Prompt in this folder, using `venv\Scripts\python.exe server.py ...`):
`--setup`, `--check`, `--add-project <path>`, `--remove-project <path>`, `--list-projects`,
`--set-check "<project folder name>" tests "<command>"`, `--install-config`, `--list-models` (double-click `list_models.bat`: shows which model names your keys can use).

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| ai_workforce not "running" | run `check.bat`; re-run `setup_and_run.bat` (close Claude first) |
| "not in the allowed list" | double-click `add_project.bat` |
| "All models failed ... no API key" | re-run `setup_and_run.bat` and paste keys |
| 429 / quota errors | the server rotates keys/models and waits up to 30 s; add a second Gemini key |
| Groq says request too large | Groq free tier is tiny; Gemini does the big jobs. Keep tasks small or add Gemini keys |
| "FAILED ... Nothing was changed" | normal safety stop; ask for a smaller task, give files, or use worker "deep" |
| Wrong change applied | tell Claude "undo that" (`changes undo`) |
| 503 "high demand" / very slow answers | normal Gemini congestion: the server parks that model for 1 minute and uses the next one |
| Groq says model does not exist | run `list_models.bat`, then set `"grok_model": "<name>"` in `aiw_config.json` |
| Models retired/renamed | set `flash_models` / `pro_models` in `aiw_config.json`, or just wait: retired ones are skipped automatically |

## 7. Honest limitations

- Workers are single-shot code generators; the server adds retries, review and test-fix loops, but this is not a
  fully autonomous IDE agent. Big refactors work best as several small tasks.
- Workers see only the files selected for the task (plus a project map). Name files when you know them.
- No browser/UI testing; `run_check` only runs the commands you approve.
- `.gitignore` negation rules (`!file`) are not supported (those files stay ignored).
- Tested with simulated models. Real API behaviour (especially new-style Gemini `AQ.` keys) is verified only when you run `check.bat`.
