"""
Unstoppable AI Workforce - MCP Server v3 (project engine)
==========================================================
Claude = Manager.  Gemini Flash / Gemini Pro / Grok-or-Groq = Workers.

What v3 adds over v2
  * Works on YOUR LOCAL FOLDERS: open_project(path) - no upload, no GitHub. The server reads the
    files itself and hands the relevant ones to the workers. Claude never reads project code.
  * Two ways in:  open_project(new="name")  = fresh project,
                  open_project(project="C:\\path\\or name") = existing local project.
  * ask() / review() / work() / swarm(): understand code, audit the whole project in parallel
    batches, edit existing files with search/replace blocks (cheap, safe), create new files.
  * Safety: allow-listed folders only, secrets files never sent, secret-looking values redacted,
    every change is backed up (changes undo), proposals (apply=False), run_check only runs commands
    the user approved.
  * Auto-quality loop on the server side: syntax checks, truncation/elision detection, retries,
    optional review + auto-fix, run_check(fix=True) test-fix loop. Failed attempts cost worker
    tokens, not Claude tokens.

Kept from v1/v2: random.shuffle key load balancing, automatic cross-model fallback, retired-model
auto-skip, parallel execution (ThreadPoolExecutor), status report.

NOTE: stdout is reserved for the MCP stdio transport. All logging goes to stderr.
CLI:  python server.py --setup | --check | --add-project [path] | --list-projects |
      --remove-project <path> | --set-check <project> <name> "<cmd>" | --install-config
"""

import ast
import difflib
import fnmatch
import getpass
import hashlib
import json
import os
import posixpath
import random
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:  # MCP Python SDK 2.x renamed FastMCP -> MCPServer
    from mcp.server.mcpserver import MCPServer as _MCPServer
except ImportError:  # MCP Python SDK 1.x
    from mcp.server.fastmcp import FastMCP as _MCPServer

__version__ = "3.0.0"
mcp = _MCPServer("ai_workforce")

HERE = Path(__file__).resolve().parent


# ===========================================================================
# Configuration (env vars override aiw_config.json; the file is re-read when it changes)
# ===========================================================================
def _config_path():
    return Path(os.environ.get("AIW_CONFIG") or (HERE / "aiw_config.json"))


def data_dir():
    return Path(os.environ.get("AIW_DATA") or (HERE / ".aiw_data"))


def _read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception:
        return default


_cfg_cache = {"path": None, "mtime": "unset", "data": {}}


def _cfg_data():
    p = _config_path()
    try:
        m = p.stat().st_mtime_ns
    except OSError:
        m = None
    if _cfg_cache["path"] != str(p) or _cfg_cache["mtime"] != m:
        data = _read_json(p, {}) if m is not None else {}
        _cfg_cache.update(path=str(p), mtime=m, data=data if isinstance(data, dict) else {})
    return _cfg_cache["data"]


def cfg(key, env=None, default=""):
    if env:
        v = os.environ.get(env)
        if v is not None and v.strip() != "":
            return v.strip()
    v = _cfg_data().get(key)
    return default if v in (None, "") else v


def cfg_int(key, env, default):
    try:
        return int(cfg(key, env, default))
    except (TypeError, ValueError):
        return default


def cfg_bool(key, env, default=False):
    v = cfg(key, env, default)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on", "y")


def cfg_models(key, env, default):
    v = cfg(key, env, None)
    if v is None:
        return list(default)
    items = [str(x).strip() for x in v] if isinstance(v, list) else [x.strip() for x in str(v).split(",")]
    items = [x for x in items if x]
    return items or list(default)


def save_config(update):
    """Merge `update` into aiw_config.json (file only, never env). Returns the new dict."""
    p = _config_path()
    data = _read_json(p, {}) if p.exists() else {}
    if not isinstance(data, dict):
        data = {}
    data.update(update)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    if os.name != "nt":
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass
    _cfg_cache["mtime"] = "unset"
    return data


def workspace_dir():
    return Path(cfg("workspace", "AIW_WORKSPACE", HERE / "workspace")).expanduser().resolve()


def log(msg):
    print(f"[ai_workforce] {msg}", file=sys.stderr, flush=True)


FLASH_DEFAULT = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-2.5-flash"]
PRO_DEFAULT = ["gemini-3.1-pro-preview", "gemini-2.5-pro"]
GROK_DEFAULT = ["grok-4.3", "grok-4"]
GROQ_DEFAULT = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]


def xai_base_url():
    return str(cfg("xai_base_url", "XAI_BASE_URL", "https://api.x.ai/v1"))


def flash_models():
    return cfg_models("flash_models", "GEMINI_FLASH_MODEL", FLASH_DEFAULT)


def pro_models():
    return cfg_models("pro_models", "GEMINI_PRO_MODEL", PRO_DEFAULT)


def grok_models():
    default = GROQ_DEFAULT if "groq" in xai_base_url().lower() else GROK_DEFAULT
    return cfg_models("grok_model", "GROK_MODEL", default)


def request_timeout():
    return float(cfg("timeout_seconds", "AIW_TIMEOUT_SECONDS", 150))


def max_fixes():
    return cfg_int("max_fixes", "AIW_MAX_FIXES", 2)


MAX_SWARM_TASKS = 3
MAX_RATE_WAIT = 30.0          # seconds we are willing to wait out a rate limit
MAX_FILE_BYTES = 1_000_000    # bigger files are not indexed
MAX_FILES = 6000
TEXT_PREVIEW_CAP = 200        # max lines returned by view()

WORKER_SYSTEM_PROMPT = (
    "You are an expert worker in an AI development agency. "
    "Output ONLY the raw code/text in markdown blocks. No explanations."
)


def _valid(value):
    if not value:
        return False
    v = str(value).strip()
    return bool(v) and not v.upper().startswith("YOUR_") and not v.lower().startswith("paste")


def gemini_keys():
    pairs = [
        ("GeminiKeyA", cfg("gemini_key_a", "GEMINI_API_KEY_A", "")),
        ("GeminiKeyB", cfg("gemini_key_b", "GEMINI_API_KEY_B", "")),
        ("GeminiKey", cfg("gemini_key", "GEMINI_API_KEY", "")),
    ]
    return [(label, str(k).strip()) for label, k in pairs if _valid(k)]


def grok_keys():
    k = cfg("xai_api_key", "XAI_API_KEY", "")
    return [("GrokKey", str(k).strip())] if _valid(k) else []


# ===========================================================================
# Provider layer: call(model, key, prompt, system) -> Result
# ===========================================================================
class Result(str):
    """Worker text plus metadata (truncated flag, which model answered)."""
    truncated = False
    handled_by = ""


def _mk(text, truncated=False):
    r = Result(text)
    r.truncated = bool(truncated)
    return r


_GEM_BACKEND = {}   # api key -> "developer" | "vertex" (learned at runtime)


def _err_code(e):
    c = getattr(e, "status_code", None) or getattr(e, "code", None)
    return c if isinstance(c, int) else None


def _is_keytype_error(e):
    s = str(e).lower()
    return _err_code(e) in (400, 401, 403) or any(
        n in s for n in ("api key not valid", "api_key_invalid", "access_token_type_unsupported",
                         "unauthenticated", "permission_denied", "invalid api key"))


def _gemini_generate(backend, model_name, api_key, prompt, system):
    from google import genai
    from google.genai import types

    http = types.HttpOptions(timeout=int(request_timeout() * 1000))
    if backend == "vertex":
        client = genai.Client(vertexai=True, api_key=api_key, http_options=http)
    else:
        client = genai.Client(api_key=api_key, http_options=http)
    conf = {"system_instruction": system or WORKER_SYSTEM_PROMPT}
    mot = cfg_int("max_output_tokens", "AIW_MAX_OUTPUT_TOKENS", 0)
    if mot:
        conf["max_output_tokens"] = mot
    response = client.models.generate_content(
        model=model_name, contents=prompt, config=types.GenerateContentConfig(**conf))
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Empty response from Gemini")
    cands = getattr(response, "candidates", None) or []
    finish = str(getattr(cands[0], "finish_reason", "")) if cands else ""
    return _mk(text, "MAX_TOKENS" in finish.upper())


def call_gemini(model_name, api_key, prompt, system=None):
    """Tries the normal Gemini API first; if the key type is rejected (new AQ.-style keys), the
    Vertex AI express endpoint. The working backend is remembered per key."""
    mode = str(cfg("gemini_backend", "AIW_GEMINI_BACKEND", "auto")).lower()
    if mode in ("developer", "vertex"):
        order = [mode]
    elif api_key in _GEM_BACKEND:
        order = [_GEM_BACKEND[api_key]]
    else:
        order = ["developer", "vertex"]
    errors = []
    for backend in order:
        try:
            out = _gemini_generate(backend, model_name, api_key, prompt, system)
            _GEM_BACKEND[api_key] = backend
            return out
        except Exception as e:
            errors.append((backend, e))
            if mode == "auto" and len(order) > 1 and _is_keytype_error(e):
                continue
            raise
    first, last = errors[0][1], errors[-1][1]
    raise RuntimeError(f"{str(first)[:70]} | via vertex: {str(last)[:70]}")


def call_grok(model_name, api_key, prompt, system=None):
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=xai_base_url(), timeout=request_timeout(), max_retries=0)
    kwargs = {}
    mot = cfg_int("max_output_tokens", "AIW_MAX_OUTPUT_TOKENS", 0)
    if mot:
        kwargs["max_tokens"] = mot
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": system or WORKER_SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        **kwargs,
    )
    choice = response.choices[0]
    text = choice.message.content
    if not text:
        raise RuntimeError("Empty response from Grok")
    return _mk(text, getattr(choice, "finish_reason", "") == "length")


MODELS = {
    "flash": {"name": "Gemini Flash", "provider": "gemini", "keys": gemini_keys,
              "candidates": flash_models, "call": call_gemini},
    "pro": {"name": "Gemini Pro", "provider": "gemini", "keys": gemini_keys,
            "candidates": pro_models, "call": call_gemini},
    "grok": {"name": "Grok", "provider": "grok", "keys": grok_keys,
             "candidates": grok_models, "call": call_grok},
}

CHAINS = {
    "fast": ["flash", "pro", "grok"],
    "deep": ["pro", "flash", "grok"],
    "creative": ["grok", "pro", "flash"],
}


def model_limit(model_id):
    """Max prompt size (characters) we send to a model. Groq's free tier has tiny per-minute limits."""
    if MODELS[model_id]["provider"] == "gemini":
        return cfg_int("gemini_max_chars", "AIW_GEMINI_MAX_CHARS", 400_000)
    default = 20_000 if "groq" in xai_base_url().lower() else 300_000
    return cfg_int("grok_max_chars", "AIW_GROK_MAX_CHARS", default)


def max_prompt_chars(worker):
    """Largest prompt any usable model in this worker's chain accepts."""
    best = 0
    for mid in CHAINS[worker]:
        try:
            if MODELS[mid]["keys"]():
                best = max(best, model_limit(mid))
        except Exception:
            pass
    return best or min(model_limit(m) for m in CHAINS[worker])


# --- session health tracking -------------------------------------------------
_CLOCK = {"offset": 0.0}


def _now():
    return time.monotonic() + _CLOCK["offset"]


def _sleep(seconds):
    time.sleep(seconds)


_health_lock = threading.Lock()
_dead_models = set()      # (model_id, candidate): retired / not found
_dead_keys = set()        # (model_id, candidate, key_label): quota "limit: 0" on this key
_cool = {}                # (model_id, candidate, key_label) -> until
_bad_keys = {}            # (provider, key_label) -> until   (invalid key)


def _is_rate_limit(e):
    s = str(e).lower()
    return _err_code(e) == 429 or any(n in s for n in ("429", "rate limit", "rate_limit", "resource_exhausted",
                                                         "quota exceeded", "too many requests"))


def _is_dead_quota(e):
    s = str(e).lower()
    return "limit: 0" in s or "limit:0" in s


def _is_overloaded(e):
    s = str(e).lower()
    return _err_code(e) in (500, 502, 503, 504) or any(n in s for n in (
        "503", "unavailable", "high demand", "overloaded", "timed out", "timeout", "deadline"))


def _is_invalid_key(e):
    s = str(e).lower()
    return _err_code(e) == 401 or any(n in s for n in ("api key not valid", "invalid api key", "incorrect api key",
                                                         "invalid_api_key", "unauthorized"))


def _is_model_error(e):
    if _err_code(e) == 404:
        return True
    s = str(e).lower()
    needles = ("not found", "no longer available", "does not exist", "deprecated", "decommission",
               "is not supported for", "model_not_found", "has been retired")
    return any(n in s for n in needles)


def _retry_delay(e):
    s = str(e)
    m = re.search(r"try again in ([\d.]+)\s*(ms|s|m)", s, re.I)
    if m:
        v = float(m.group(1))
        return v / 1000 if m.group(2).lower() == "ms" else (v * 60 if m.group(2).lower() == "m" else v)
    m = re.search(r"retry(?:Delay)?\W{1,6}(\d+(?:\.\d+)?)s", s, re.I)
    return float(m.group(1)) if m else None


def reset_health():
    with _health_lock:
        _dead_models.clear()
        _dead_keys.clear()
        _cool.clear()
        _bad_keys.clear()


# --- status log ----------------------------------------------------------------
_status_log = []
_status_lock = threading.Lock()


def record(worker, label, handled_by, key_label, seconds, ok, fallbacks):
    with _status_lock:
        _status_log.append({"n": len(_status_log) + 1, "worker": worker,
                            "task": " ".join(str(label).split())[:40], "handled_by": handled_by,
                            "key": key_label, "seconds": seconds, "ok": ok, "fallbacks": fallbacks})


def _try_chain(worker, prompt, system, size, failures):
    """One pass over the fallback chain. Returns (Result|None, shortest_wait|None)."""
    waits = []
    for model_id in CHAINS[worker]:
        spec = MODELS[model_id]
        if size > model_limit(model_id):
            failures.append(f"{spec['name']}: prompt too large ({size} > {model_limit(model_id)} chars)")
            continue
        try:
            keys = list(spec["keys"]())
        except Exception as e:
            failures.append(f"{spec['name']}: key lookup failed ({e})")
            continue
        if not keys:
            failures.append(f"{spec['name']}: no API key configured")
            continue
        random.shuffle(keys)  # load balancing: fresh random key order for every call
        for candidate in spec["candidates"]():
            with _health_lock:
                if (model_id, candidate) in _dead_models:
                    continue
            for key_label, key in keys:
                with _health_lock:
                    if (model_id, candidate, key_label) in _dead_keys:
                        continue
                    if _bad_keys.get((spec["provider"], key_label), 0) > _now():
                        continue
                    cd = _cool.get((model_id, candidate, key_label), 0)
                    if cd > _now():
                        waits.append(cd - _now())
                        continue
                try:
                    out = spec["call"](candidate, key, prompt, system or WORKER_SYSTEM_PROMPT)
                    res = out if isinstance(out, Result) else _mk(str(out))
                    res.handled_by = f"{spec['name']} ({candidate})"
                    res.key_label = key_label
                    return res, None
                except Exception as e:
                    msg = str(e).replace("\n", " ")[:140]
                    failures.append(f"{spec['name']}/{candidate}/{key_label}: {msg}")
                    log(f"Failure -> {spec['name']} {candidate} ({key_label}): {msg}")
                    with _health_lock:
                        if _is_rate_limit(e) and not _is_dead_quota(e):
                            delay = _retry_delay(e) or 20.0
                            _cool[(model_id, candidate, key_label)] = _now() + delay
                            waits.append(delay)
                        elif _is_dead_quota(e):
                            _dead_keys.add((model_id, candidate, key_label))
                        elif _is_invalid_key(e):
                            _bad_keys[(spec["provider"], key_label)] = _now() + 600
                        elif _is_overloaded(e):
                            # the model itself is busy/slow: park it for a minute (all keys), try the next one
                            for kl, _k in keys:
                                _cool[(model_id, candidate, kl)] = _now() + 60
                            waits.append(60.0)
                            break
                        elif _is_model_error(e):
                            _dead_models.add((model_id, candidate))
                            log(f"Model {candidate} looks retired; skipping it from now on.")
                            break  # other keys will not help for a dead model id
                        elif _err_code(e) == 403:
                            _cool[(model_id, candidate, key_label)] = _now() + 600
    return None, (min(waits) if waits else None)


def run_worker(worker, prompt, system=None, label=None):
    """Runs a prompt through the worker's fallback chain (shuffled keys, cooldown aware).
    Returns a Result on success or a plain 'ERROR: ...' string. Never raises."""
    label = label or prompt
    start = time.time()
    failures = []
    try:
        size = len(prompt) + len(system or WORKER_SYSTEM_PROMPT)
        for round_ in range(2):
            res, wait = _try_chain(worker, prompt, system, size, failures)
            if res is not None:
                record(worker, label, res.handled_by, getattr(res, "key_label", "-"),
                       round(time.time() - start, 2), True, len(failures))
                return res
            if round_ == 0 and wait is not None and wait <= MAX_RATE_WAIT:
                log(f"Rate limited everywhere; waiting {wait:.0f}s then retrying once.")
                _sleep(wait + 0.5)
                continue
            break
        record(worker, label, "NONE (all failed)", "-", round(time.time() - start, 2), False, len(failures))
        return "ERROR: All models in the fallback chain failed.\n" + "\n".join(f"- {f}" for f in failures[-8:])
    except Exception as e:  # never crash the MCP server
        record(worker, label, "INTERNAL ERROR", "-", round(time.time() - start, 2), False, len(failures))
        return f"ERROR: Unexpected failure: {e}"


# ===========================================================================
# Text helpers: secrets, sensitive files, encodings, globs
# ===========================================================================
PLACEHOLDER = "<<REDACTED_SECRET>>"

_SECRET_RES = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"\bAQ\.[0-9A-Za-z_\-]{30,}"),
    re.compile(r"\bgsk_[0-9A-Za-z]{30,}"),
    re.compile(r"\bxai-[0-9A-Za-z]{30,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
]
_SECRET_ASSIGN_RE = re.compile(
    r"""(?i)((?:password|passwd|secret|token|api[_-]?key|apikey|private[_-]?key)["']?\s*[:=]\s*["'])([^"'\n]{8,})(["'])""")


def redact(text):
    """Replaces secret-looking values with a placeholder. Returns (text, count)."""
    n = 0
    for rx in _SECRET_RES:
        text, k = rx.subn(PLACEHOLDER, text)
        n += k

    def _sub(m):
        nonlocal n
        if PLACEHOLDER in m.group(2):
            return m.group(0)
        n += 1
        return m.group(1) + PLACEHOLDER + m.group(3)

    text = _SECRET_ASSIGN_RE.sub(_sub, text)
    return text, n


SENSITIVE_PATTERNS = [".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "id_ed25519*", "id_dsa*",
                      "*.keystore", "*.jks", "credentials*.json", "service-account*.json", "serviceaccount*.json",
                      "secrets.*", "*.secret", "*.kdbx", ".npmrc", ".pypirc", ".netrc", "aiw_config.json",
                      "google-services.json", "*.ppk"]
SENSITIVE_OK = {".env.example", ".env.sample", ".env.template", ".env.dist", ".env.defaults"}


def is_sensitive(rel):
    base = rel.rsplit("/", 1)[-1].lower()
    if base in SENSITIVE_OK:
        return False
    return any(fnmatch.fnmatch(base, p) for p in SENSITIVE_PATTERNS)


IGNORE_DIRS = {".git", ".hg", ".svn", "node_modules", "venv", ".venv", "env", "__pycache__", ".mypy_cache",
               ".pytest_cache", ".ruff_cache", ".tox", "dist", "build", ".next", ".nuxt", ".svelte-kit", "target",
               ".idea", ".vscode", ".gradle", ".dart_tool", "coverage", ".cache", "bower_components", "vendor",
               "site-packages", ".terraform", "Pods", "DerivedData", ".parcel-cache", ".turbo", ".expo"}
IGNORE_FILES = {".DS_Store", "Thumbs.db", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
                "Pipfile.lock", "composer.lock", "Cargo.lock", "gradle.lockfile"}
BIN_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg", ".pdf", ".zip", ".gz", ".tar", ".rar",
           ".7z", ".exe", ".dll", ".so", ".dylib", ".class", ".jar", ".war", ".pyc", ".pyo", ".o", ".a", ".lib",
           ".mp3", ".mp4", ".mov", ".avi", ".wav", ".ogg", ".woff", ".woff2", ".ttf", ".otf", ".eot", ".db",
           ".sqlite", ".sqlite3", ".apk", ".aab", ".ipa", ".bin", ".dat", ".pickle", ".pkl", ".npy", ".onnx",
           ".pt", ".h5", ".psd", ".ai", ".map", ".min.js", ".min.css", ".lockb"}
LANG_BY_EXT = {
    ".py": "python", ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js", ".ts": "ts", ".tsx": "ts",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".php": "php", ".cs": "csharp", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".ino": "cpp", ".swift": "swift", ".dart": "dart", ".scala": "scala", ".sql": "sql", ".sh": "shell",
    ".bat": "batch", ".ps1": "powershell", ".vue": "vue", ".svelte": "svelte", ".lua": "lua",
    ".html": "html", ".htm": "html", ".mako": "html", ".jinja": "html", ".jinja2": "html", ".css": "css",
    ".scss": "css", ".less": "css",
}
CODE_LANGS = {"python", "js", "ts", "java", "kotlin", "go", "rust", "ruby", "php", "csharp", "c", "cpp", "swift",
              "dart", "scala", "sql", "shell", "batch", "powershell", "vue", "svelte", "lua", "html", "css"}


def lang_of(rel):
    ext = os.path.splitext(rel)[1].lower()
    return LANG_BY_EXT.get(ext, "other")


def read_text_file(p):
    """Returns (text_with_LF, eol, bom, raw_bytes) or None for binary/undecodable files."""
    data = Path(p).read_bytes()
    if b"\x00" in data[:4096]:
        return None
    bom = data.startswith(b"\xef\xbb\xbf")
    try:
        text = data.decode("utf-8-sig" if bom else "utf-8")
    except UnicodeDecodeError:
        return None
    crlf = text.count("\r\n")
    eol = "\r\n" if crlf and crlf >= text.count("\n") - crlf else "\n"
    return text.replace("\r\n", "\n"), eol, bom, data


def encode_text(text, eol="\n", bom=False):
    return (("\ufeff" if bom else "") + text.replace("\n", eol)).encode("utf-8")


def sha(data):
    return hashlib.sha1(data if data is not None else b"\x00none").hexdigest()


def _norm_rel(p):
    p = str(p).strip().strip('"').strip("'").replace("\\", "/")
    p = re.sub(r"^(\./)+", "", p)
    return re.sub(r"/{2,}", "/", p)


def glob_to_re(pat):
    i, out = 0, ""
    while i < len(pat):
        c = pat[i]
        if c == "*":
            if pat[i:i + 2] == "**":
                i += 2
                if pat[i:i + 1] == "/":
                    i += 1
                    out += "(?:.*/)?"
                else:
                    out += ".*"
                continue
            out += "[^/]*"
        elif c == "?":
            out += "[^/]"
        else:
            out += re.escape(c)
        i += 1
    return re.compile("^" + out + "$", re.I)


class GitIgnore:
    """Small .gitignore reader (no negation support)."""

    def __init__(self, text=""):
        self.rules = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            dir_only = line.endswith("/")
            pat = line.rstrip("/")
            anchored = "/" in pat
            pat = pat.lstrip("/")
            if not pat:
                continue
            self.rules.append((glob_to_re(pat) if anchored else None, pat, anchored, dir_only))

    def match(self, rel, is_dir):
        base = rel.rsplit("/", 1)[-1]
        for rx, pat, anchored, dir_only in self.rules:
            if dir_only and not is_dir:
                continue
            if anchored:
                if rx.match(rel):
                    return True
            elif fnmatch.fnmatch(base.lower(), pat.lower()):
                return True
        return False


def _within(child, root):
    c = os.path.normcase(os.path.realpath(str(child)))
    r = os.path.normcase(os.path.realpath(str(root)))
    return c == r or c.startswith(r.rstrip(os.sep) + os.sep)


# ===========================================================================
# Code index: symbols, imports, stack detection
# ===========================================================================
_SYM = {
    "js": [r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\*?\s+([A-Za-z_$][\w$]*)",
           r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)",
           r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=>",
           r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:React\.)?(?:memo|forwardRef|createContext)\("],
    "java": [r"^\s*(?:public\s+|protected\s+|private\s+|abstract\s+|final\s+|static\s+)*(?:class|interface|enum|record)\s+(\w+)",
             r"^\s*(?:public|protected|private)\s+(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?[\w<>\[\],.? ]+?\s+(\w+)\s*\("],
    "kotlin": [r"\bfun\s+(?:<[^>]+>\s*)?(?:[\w.]+\.)?(\w+)\s*\(", r"^\s*(?:data\s+|sealed\s+|abstract\s+|open\s+)?(?:class|object|interface)\s+(\w+)"],
    "go": [r"^func\s+(?:\([^)]*\)\s*)?(\w+)", r"^type\s+(\w+)\s+(?:struct|interface)"],
    "rust": [r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)", r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)"],
    "ruby": [r"^\s*def\s+(?:self\.)?(\w+[?!]?)", r"^\s*(?:class|module)\s+(\w+)"],
    "php": [r"\bfunction\s+(\w+)\s*\(", r"^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait)\s+(\w+)"],
    "csharp": [r"^\s*(?:public\s+|internal\s+|private\s+|protected\s+|static\s+|abstract\s+|sealed\s+|partial\s+)*(?:class|interface|struct|enum|record)\s+(\w+)",
               r"^\s*(?:public|private|protected|internal)\s+(?:static\s+)?(?:async\s+)?(?:override\s+)?[\w<>\[\],?]+\s+(\w+)\s*\("],
    "cpp": [r"^\s*(?:class|struct)\s+(\w+)", r"^[A-Za-z_][\w:<>\*&\s]*?\b(\w+)\s*\([^;{)]*\)\s*(?:const)?\s*\{"],
    "swift": [r"\bfunc\s+(\w+)", r"^\s*(?:class|struct|enum|protocol|extension)\s+(\w+)"],
    "dart": [r"^\s*(?:abstract\s+)?class\s+(\w+)"],
    "scala": [r"\bdef\s+(\w+)", r"^\s*(?:case\s+)?(?:class|object|trait)\s+(\w+)"],
    "sql": [r"(?i)create\s+(?:or\s+replace\s+)?(?:table|view|function|procedure|trigger)\s+(?:if\s+not\s+exists\s+)?([\w.\"]+)"],
    "shell": [r"^\s*(?:function\s+)?(\w+)\s*\(\)\s*\{"],
    "lua": [r"^\s*(?:local\s+)?function\s+([\w.:]+)"],
}
_SYM["ts"] = _SYM["js"] + [r"^\s*(?:export\s+)?(?:declare\s+)?(?:interface|type|enum)\s+([A-Za-z_$][\w$]*)"]
_SYM["vue"] = _SYM["svelte"] = _SYM["js"]
_SYM_RE = {k: [re.compile(p, re.M) for p in v] for k, v in _SYM.items()}
_JS_IMPORT_RE = re.compile(r"""(?:\bfrom\s+|\bimport\s+|\brequire\(\s*|\bimport\(\s*)["']([^"']+)["']""")


def parse_source(rel, text):
    """Returns (symbols, imports) for a source file."""
    lang = lang_of(rel)
    syms, imps = [], []
    if lang == "python":
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            tree = None
        if tree is not None:
            for n in tree.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    syms.append(n.name)
                    if isinstance(n, ast.ClassDef):
                        syms.extend(f"{n.name}.{m.name}" for m in n.body
                                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                                    and not m.name.startswith("_"))
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    imps.extend([0, a.name, []] for a in n.names)
                elif isinstance(n, ast.ImportFrom):
                    imps.append([n.level or 0, n.module or "", [a.name for a in n.names]])
        else:
            syms = re.findall(r"^\s*(?:async\s+)?(?:def|class)\s+(\w+)", text, re.M)
    elif lang in _SYM_RE:
        seen = set()
        for rx in _SYM_RE[lang]:
            for m in rx.finditer(text):
                name = m.group(1)
                if name and name not in seen and name not in ("if", "for", "while", "switch", "catch", "return"):
                    seen.add(name)
                    syms.append(name)
        if lang in ("js", "ts", "vue", "svelte"):
            imps = [[0, s, []] for s in _JS_IMPORT_RE.findall(text)]
    return syms[:80], imps[:80]


def resolve_import(files, rel, lang, imp):
    """Maps one import record to project files (best effort)."""
    base_dir = rel.rsplit("/", 1)[0] if "/" in rel else ""
    level, mod, names = imp
    cands = []
    if lang == "python":
        if level:
            d = base_dir
            for _ in range(level - 1):
                d = d.rsplit("/", 1)[0] if "/" in d else ""
            base = (d + "/" if d else "") + mod.replace(".", "/") if mod else d
            if mod:
                cands += [base + ".py", base + "/__init__.py"]
            for n in names:
                cands += [(base + "/" if base else "") + n + ".py"]
        else:
            p = mod.replace(".", "/")
            top = base_dir.split("/")[0] + "/" if base_dir else ""
            for prefix in ("", "src/", top):
                cands += [prefix + p + ".py", prefix + p + "/__init__.py"]
    elif lang in ("js", "ts", "vue", "svelte"):
        spec = mod
        if spec.startswith("."):
            joined = posixpath.normpath(posixpath.join(base_dir, spec))
        elif spec.startswith("@/") or spec.startswith("~/"):
            joined = "src/" + spec[2:]
        else:
            return []
        cands.append(joined)
        for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".json", ".vue", ".svelte"):
            cands.append(joined + ext)
        for ext in (".ts", ".tsx", ".js", ".jsx"):
            cands.append(joined + "/index" + ext)
    return [c for c in dict.fromkeys(cands) if c in files]


_STOP = set("""the a an and or to of in on for with from this that these those it its is are be as at by add make fix
change update create build write new use using please code file files function functions class method methods
project app all any some your my our can should need want also into out not but then than when where which what
how why do does did get set run""".split())

TEST_RE = re.compile(r"(^|/)(tests?|__tests__|spec|e2e|cypress)(/|$)|(^|/)(test_[^/]*|[^/]*(_test|\.test|\.spec)\.[A-Za-z]+)$", re.I)
ENTRY_RE = re.compile(r"(^|/)(main|app|index|server|manage|wsgi|asgi|__main__)\.(py|js|ts|tsx|jsx|java|kt|go|rs|php|rb|cs|dart)$|(^|/)(MainActivity|Application)\.(java|kt)$", re.I)
RISK_WORDS = ("auth", "login", "password", "token", "session", "payment", "admin", "api", "route", "router", "db",
              "database", "sql", "upload", "crypto", "secret", "user", "otp", "jwt", "cors", "middleware",
              "security", "permission", "webhook", "email", "supabase", "config", "settings")


def search_terms(text):
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text or "")
    out = []
    for w in words:
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", w).replace("_", " ").split()
        for p in [w] + parts:
            pl = p.lower()
            if len(pl) >= 3 and pl not in _STOP and pl not in out:
                out.append(pl)
    return out[:40]


class ProjectError(Exception):
    pass


# ===========================================================================
# Project
# ===========================================================================
class Project:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.name = self.root.name or "project"
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", self.name)[:30]
        self.pid = f"{slug}-{hashlib.sha1(os.path.normcase(str(self.root)).encode()).hexdigest()[:6]}"
        self.data = data_dir() / self.pid
        self.lock = threading.RLock()
        self.files = {}
        self.ignored = 0
        self.truncated = False
        self.refresh()

    # ----------------------------------------------------------------- index
    def refresh(self):
        with self.lock:
            old, new = self.files, {}
            gi_path = self.root / ".gitignore"
            gi = GitIgnore(gi_path.read_text(encoding="utf-8", errors="replace") if gi_path.is_file() else "")
            ignored, truncated = 0, False
            for dirpath, dirnames, filenames in os.walk(self.root):
                reld = os.path.relpath(dirpath, self.root).replace(os.sep, "/")
                reld = "" if reld == "." else reld
                keep = []
                for d in sorted(dirnames):
                    rp = f"{reld}/{d}" if reld else d
                    if d in IGNORE_DIRS or d.startswith(".aiw") or gi.match(rp, True) \
                            or os.path.islink(os.path.join(dirpath, d)):
                        ignored += 1
                        continue
                    keep.append(d)
                dirnames[:] = keep
                for fn in sorted(filenames):
                    rp = f"{reld}/{fn}" if reld else fn
                    full = os.path.join(dirpath, fn)
                    if fn in IGNORE_FILES or gi.match(rp, False) or os.path.islink(full):
                        ignored += 1
                        continue
                    try:
                        st = os.stat(full)
                    except OSError:
                        continue
                    ext = os.path.splitext(fn)[1].lower()
                    ent = {"size": st.st_size, "mt": st.st_mtime_ns, "sens": is_sensitive(rp),
                           "lang": lang_of(rp)}
                    prev = old.get(rp)
                    if ent["sens"]:
                        ent["bin"] = True
                    elif ext in BIN_EXT or fn.lower().endswith((".min.js", ".min.css")) or st.st_size > MAX_FILE_BYTES:
                        ent["bin"] = True
                    elif prev and prev.get("size") == st.st_size and prev.get("mt") == st.st_mtime_ns:
                        ent = prev
                    else:
                        try:
                            got = read_text_file(full)
                        except OSError:
                            got = None
                        if got is None:
                            ent["bin"] = True
                        else:
                            text = got[0]
                            ent["lines"] = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
                            ent["eol"] = got[1]
                            syms, imps = parse_source(rp, text)
                            ent["sym"] = syms
                            ent["imp"] = imps
                    new[rp] = ent
                    if len(new) >= MAX_FILES:
                        truncated = True
                        break
                if truncated:
                    break
            self.files, self.ignored, self.truncated = new, ignored, truncated

    def is_text(self, rel):
        e = self.files.get(rel)
        return bool(e) and not e.get("bin")

    def text_files(self):
        return [r for r, e in self.files.items() if not e.get("bin")]

    def canon(self, rel):
        rel = _norm_rel(rel)
        if rel in self.files:
            return rel
        low = rel.lower()
        for r in self.files:
            if r.lower() == low:
                return r
        return rel

    def resolve(self, rel):
        rel = _norm_rel(rel)
        if not rel:
            raise ProjectError("empty path")
        if rel.startswith("/") or rel.startswith("~") or re.match(r"^[A-Za-z]:", rel):
            raise ProjectError(f"path must be relative to the project folder: {rel}")
        if ".." in rel.split("/"):
            raise ProjectError(f"path escapes the project folder: {rel}")
        real = Path(os.path.realpath(self.root / rel))
        if not _within(real, self.root):
            raise ProjectError(f"path is outside the project folder: {rel}")
        return real

    def write_block_reason(self, rel):
        parts = rel.split("/")
        low = [p.lower() for p in parts]
        for p in low[:-1]:
            if p in IGNORE_DIRS or p.startswith(".aiw"):
                return "inside a protected/ignored folder"
        if is_sensitive(rel):
            return "secrets file (never touched)"
        return None

    def expand(self, patterns):
        """-> (existing text files, unmatched patterns)"""
        out, unmatched = [], []
        low = {r.lower(): r for r in self.files}
        for pat in patterns or []:
            pat = _norm_rel(pat)
            if not pat:
                continue
            hits = []
            if any(c in pat for c in "*?"):
                rx = glob_to_re(pat)
                hits = [r for r in self.files if rx.match(r)]
            elif pat in self.files:
                hits = [pat]
            elif pat.lower() in low:
                hits = [low[pat.lower()]]
            else:
                pref = pat.rstrip("/").lower() + "/"
                hits = [r for r in self.files if r.lower().startswith(pref)]
            hits = [h for h in hits if self.is_text(h)]
            if hits:
                out.extend(hits)
            else:
                unmatched.append(pat)
        return list(dict.fromkeys(out)), unmatched

    # --------------------------------------------------------------- reading
    def read(self, rel):
        p = self.resolve(rel)
        if not p.is_file():
            return None
        return read_text_file(p)

    def read_for_worker(self, rel, numbered=False):
        """Redacted text of a file as the workers may see it (None if not shareable)."""
        e = self.files.get(rel)
        if not e or e.get("bin"):
            return None
        got = self.read(rel)
        if got is None:
            return None
        text, _n = redact(got[0])
        if numbered:
            lines = text.split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            return "\n".join(f"{i:>4}| {l}" for i, l in enumerate(lines, 1))
        return text

    def pack(self, rels, budget, label="CONTEXT", numbered=False):
        """-> (text, packed_rels, skipped_rels)"""
        parts, packed, skipped, used = [], [], [], 0
        for rel in rels:
            t = self.read_for_worker(rel, numbered)
            if t is None:
                continue
            n = self.files[rel].get("lines", 0)
            block = f"=== {label}: {rel} ({n} lines) ===\n{t}\n=== END {label} ==="
            if used + len(block) > budget:
                skipped.append(rel)
                continue
            parts.append(block)
            packed.append(rel)
            used += len(block) + 2
        return "\n\n".join(parts), packed, skipped

    # ------------------------------------------------------------ selection
    def neighbors(self, rels, limit=8):
        out = []
        for rel in rels:
            e = self.files.get(rel) or {}
            for imp in e.get("imp", []):
                for r in resolve_import(self.files, rel, e.get("lang", ""), imp):
                    if r not in rels and r not in out and self.is_text(r):
                        out.append(r)
                    if len(out) >= limit:
                        return out
        return out

    def select(self, text, k=6, exclude=()):
        terms = search_terms(text)
        want_tests = bool(re.search(r"\btests?\b|pytest|jest|spec|unit", text or "", re.I))
        scored = []
        for rel, e in self.files.items():
            if e.get("bin") or rel in exclude:
                continue
            low = rel.lower()
            base = low.rsplit("/", 1)[-1]
            s = 0.0
            for t in terms:
                if t in base:
                    s += 4
                elif t in low:
                    s += 2
                if any(t in sy.lower() for sy in e.get("sym", ())):
                    s += 2
            if e.get("lang") not in CODE_LANGS:
                s *= 0.6
            if TEST_RE.search(rel) and not want_tests:
                s *= 0.5
            scored.append([s, rel])
        scored.sort(key=lambda x: (-x[0], x[1]))
        read_budget = 3_000_000
        for item in scored[:400]:
            rel = item[1]
            size = self.files[rel]["size"]
            if size > 200_000 or read_budget < 0:
                continue
            try:
                got = self.read(rel)
            except (OSError, ProjectError):
                continue
            if not got:
                continue
            read_budget -= size
            low = got[0].lower()
            hits = sum(min(low.count(t), 5) for t in terms)
            item[0] += hits * 0.5
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = [r for s, r in scored if s > 0][:k]
        if not top:
            top = [r for r in self.text_files() if ENTRY_RE.search(r)][:k]
        return top

    # ------------------------------------------------------------------ map
    def map_text(self, budget=10000):
        rels = sorted(r for r, e in self.files.items())
        if not rels:
            return "(empty project)"
        for level in (0, 1, 2, 3):
            lines = []
            by_dir = {}
            for r in rels:
                d = r.rsplit("/", 1)[0] if "/" in r else "."
                by_dir.setdefault(d, []).append(r)
            for d in sorted(by_dir):
                items = by_dir[d]
                if level >= 3 and len(items) > 12:
                    exts = {}
                    for r in items:
                        x = os.path.splitext(r)[1] or "(none)"
                        exts[x] = exts.get(x, 0) + 1
                    lines.append(f"{d}/ ... {len(items)} files (" + ", ".join(f"{k} {v}" for k, v in sorted(exts.items())[:4]) + ")")
                    continue
                for r in items:
                    e = self.files[r]
                    if e.get("sens"):
                        lines.append(f"{r} [secrets file, hidden]")
                    elif e.get("bin"):
                        if level <= 1:
                            lines.append(f"{r} [binary/large]")
                    else:
                        n = e.get("lines", 0)
                        syms = e.get("sym", [])
                        show = [] if level >= 2 or e.get("lang") not in CODE_LANGS else syms[:(6 if level == 0 else 3)]
                        if level >= 2 and e.get("lang") not in CODE_LANGS and not ENTRY_RE.search(r) and len(rels) > 150:
                            continue
                        lines.append(f"{r} {n}L" + (": " + ", ".join(show) if show else ""))
            text = "\n".join(lines)
            if len(text) <= budget:
                return text
        return text[:budget] + "\n... (map truncated)"

    def detect_stack(self):
        names = set(self.files)
        low_names = {r.lower() for r in names}
        stack = []

        def add(x):
            if x not in stack:
                stack.append(x)

        def has(n):
            return any(r == n or r.endswith("/" + n) for r in low_names)

        pydeps = ""
        for r in names:
            b = r.lower().rsplit("/", 1)[-1]
            if (b.startswith("requirements") and b.endswith(".txt")) or b in ("pyproject.toml", "pipfile", "setup.py"):
                try:
                    pydeps += (self.read(r) or ("",))[0].lower() + "\n"
                except (OSError, ProjectError):
                    pass
        if pydeps or any(self.files[r].get("lang") == "python" for r in names):
            py = []
            for kw, label in (("fastapi", "FastAPI"), ("flask", "Flask"), ("django", "Django"), ("sqlalchemy", "SQLAlchemy"),
                              ("supabase", "Supabase"), ("resend", "Resend"), ("pydantic", "Pydantic"), ("celery", "Celery"),
                              ("streamlit", "Streamlit"), ("pytest", "pytest")):
                if kw in pydeps:
                    py.append(label)
            if any(self.files[r].get("lang") == "python" for r in names):
                add("Python" + (" (" + ", ".join(py[:5]) + ")" if py else ""))
        jsdeps = ""
        for r in names:
            if r.lower().rsplit("/", 1)[-1] == "package.json":
                try:
                    jsdeps += (self.read(r) or ("",))[0].lower() + "\n"
                except (OSError, ProjectError):
                    pass
        if jsdeps:
            js = []
            for kw, label in (("react-native", "React Native"), ("\"next\"", "Next.js"), ("\"react\"", "React"),
                              ("\"vue\"", "Vue"), ("@angular/core", "Angular"), ("\"svelte\"", "Svelte"),
                              ("\"express\"", "Express"), ("\"vite\"", "Vite"), ("tailwindcss", "Tailwind"),
                              ("typescript", "TypeScript"), ("@supabase/supabase-js", "Supabase"), ("electron", "Electron")):
                if kw in jsdeps:
                    js.append(label)
            add("Node/JS" + (" (" + ", ".join(js[:6]) + ")" if js else ""))
        gradle = ""
        for r in names:
            b = r.lower().rsplit("/", 1)[-1]
            if b in ("pom.xml", "build.gradle", "build.gradle.kts"):
                try:
                    gradle += (self.read(r) or ("",))[0].lower()
                except (OSError, ProjectError):
                    pass
        if gradle:
            add("Java/Kotlin" + (" (Spring Boot)" if "spring-boot" in gradle else "") + (" Android" if "com.android" in gradle else ""))
        if has("androidmanifest.xml"):
            add("Android")
        if has("dockerfile") or has("docker-compose.yml") or has("docker-compose.yaml"):
            add("Docker")
        for marker, label in (("go.mod", "Go"), ("cargo.toml", "Rust"), ("pubspec.yaml", "Flutter/Dart"),
                              ("composer.json", "PHP"), ("platformio.ini", "PlatformIO/Arduino")):
            if has(marker):
                add(label)
        if any(r.endswith(".csproj") for r in low_names):
            add(".NET")
        if any(r.endswith(".ino") for r in low_names):
            add("Arduino/ESP")
        return " + ".join(stack) if stack else "unknown"

    def entry_points(self):
        return [r for r in self.text_files() if ENTRY_RE.search(r)][:5]

    def summary(self):
        text = self.text_files()
        total_lines = sum(self.files[r].get("lines", 0) for r in text)
        langs = {}
        for r in text:
            lg = self.files[r].get("lang", "other")
            langs[lg] = langs.get(lg, 0) + 1
        top = ", ".join(f"{k} {v}" for k, v in sorted(langs.items(), key=lambda x: -x[1])[:6])
        dirs = {}
        for r in text:
            d = r.split("/")[0] if "/" in r else "."
            dirs[d] = dirs.get(d, 0) + 1
        dtxt = ", ".join(f"{d}/ ({n})" if d != "." else f"(root) ({n})" for d, n in sorted(dirs.items(), key=lambda x: -x[1])[:8])
        sens = [r for r, e in self.files.items() if e.get("sens")]
        tests = [r for r in text if TEST_RE.search(r)]
        lines = [f"Opened '{self.name}' at {self.root}",
                 f"Stack: {self.detect_stack()}",
                 f"Files: {len(text)} text files, {total_lines} lines ({top or 'none'})",
                 f"Dirs: {dtxt or '(empty)'}"]
        eps = self.entry_points()
        if eps:
            lines.append("Entry: " + ", ".join(eps))
        if tests:
            lines.append(f"Tests: {len(tests)} files")
        note = []
        if sens:
            note.append(f"{len(sens)} secrets file(s) hidden from workers" + (" (" + ", ".join(sens[:3]) + ")" if sens else ""))
        if self.ignored:
            note.append(f"{self.ignored} ignored (node_modules, venv, build, .gitignore...)")
        if self.truncated:
            note.append(f"index capped at {MAX_FILES} files")
        if note:
            lines.append("Skipped: " + "; ".join(note))
        b = self.brief()
        lines.append("Brief: " + (f"{len(b)} chars saved" if b else "(none - pass brief=... to share conventions with workers)"))
        return "\n".join(lines)

    # ----------------------------------------------------------------- brief
    def brief(self):
        try:
            return (self.data / "brief.md").read_text(encoding="utf-8")[:3000]
        except OSError:
            return ""

    def set_brief(self, text):
        self.data.mkdir(parents=True, exist_ok=True)
        (self.data / "brief.md").write_text(text.strip()[:3000] + "\n", encoding="utf-8")


# ===========================================================================
# Project registry (allow-list)
# ===========================================================================
def allowed_roots():
    roots = [workspace_dir()]
    raw = os.environ.get("AIW_PROJECTS", "")
    items = [x for x in raw.split(os.pathsep) if x.strip()] if raw else []
    items += [x for x in (_cfg_data().get("allowed_roots") or []) if isinstance(x, str)]
    for it in items:
        try:
            p = Path(it.strip().strip('"')).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if p not in roots:
            roots.append(p)
    return roots


def is_too_broad(p):
    p = Path(p).resolve()
    if p.parent == p:
        return True
    try:
        return os.path.normcase(str(p)) == os.path.normcase(str(Path.home().resolve()))
    except (OSError, RuntimeError):
        return False


def add_root(path):
    p = Path(str(path).strip().strip('"').strip("'")).expanduser()
    if not p.is_absolute():
        raise ProjectError("please give the full folder path")
    p = p.resolve()
    if not p.is_dir():
        raise ProjectError(f"folder not found: {p}")
    roots = [str(Path(x).resolve()) for x in (_cfg_data().get("allowed_roots") or [])]
    if str(p) not in roots:
        roots.append(str(p))
    save_config({"allowed_roots": roots})
    return p


def remove_root(path):
    p = str(Path(str(path).strip().strip('"')).expanduser().resolve())
    roots = [r for r in (_cfg_data().get("allowed_roots") or []) if str(Path(r).resolve()) != p]
    save_config({"allowed_roots": roots})
    return p


def find_project_dir(arg):
    arg = arg.strip().strip('"').strip("'")
    roots = allowed_roots()
    looks_path = any(c in arg for c in "/\\") or re.match(r"^[A-Za-z]:", arg) or arg.startswith("~")
    if looks_path:
        p = Path(arg).expanduser()
        if not p.is_absolute():
            raise ProjectError("use the full folder path (e.g. C:\\Users\\you\\Projects\\myapp)")
        p = p.resolve()
        if not p.is_dir():
            raise ProjectError(f"folder not found: {p}")
        if not any(_within(p, r) for r in roots):
            raise ProjectError(
                f"'{p}' is not in the allowed list. Add it once (10 seconds): double-click add_project.bat "
                f"in the ai_workforce folder, or run: python server.py --add-project \"{p}\" - then try again "
                f"(no restart needed).")
        return p
    found = []
    for r in roots:
        if r.name.lower() == arg.lower() and r.is_dir():
            found.append(r)
        try:
            for c in sorted(r.iterdir()):
                if c.is_dir() and c.name.lower() == arg.lower() and c not in found:
                    found.append(c)
        except OSError:
            pass
    if not found:
        raise ProjectError(f"no allowed project named '{arg}'. Call open_project() with no arguments to list them, "
                           "or give the full folder path.")
    if len(found) > 1:
        raise ProjectError("several projects match: " + "; ".join(str(f) for f in found) + " - give the full path")
    return found[0]


def list_projects_text():
    lines = []
    for r in allowed_roots():
        if not r.is_dir():
            continue
        try:
            subs = [c.name for c in sorted(r.iterdir()) if c.is_dir() and c.name not in IGNORE_DIRS and not c.name.startswith(".")]
        except OSError:
            subs = []
        lines.append(f"{r}  ->  " + (", ".join(subs[:25]) if subs else "(no subfolders)"))
    return "Allowed folders (open one by name or full path):\n" + "\n".join(lines)


ACTIVE = {"project": None}
_active_lock = threading.Lock()


def _active():
    p = ACTIVE["project"]
    if p is None:
        raise ProjectError("no project is open. Call open_project(project=<folder path or name>) or "
                           "open_project(new=<name>) first.")
    p.refresh()
    return p


# ===========================================================================
# Edit engine: parse worker blocks, apply search/replace, validate
# ===========================================================================
EDIT_SYSTEM_PROMPT = (
    "You are an expert software engineer working inside a real project through an automated pipeline. "
    "Reply with change blocks ONLY: no prose, no markdown fences, no explanations.\n\n"
    "1) NEW file (or complete rewrite of a small file):\n"
    "=== FILE: path/relative/to/project/root ===\n"
    "<entire file content>\n"
    "=== END FILE ===\n\n"
    "2) CHANGE an existing file with search/replace edits (never retype unchanged code):\n"
    "=== EDIT: path/relative/to/project/root ===\n"
    "<<<<<<< SEARCH\n"
    "<lines copied EXACTLY from the current file; as short as possible but unique in the file>\n"
    "=======\n"
    "<replacement lines>\n"
    ">>>>>>> REPLACE\n"
    "=== END EDIT ===\n"
    "One EDIT block may hold several SEARCH/REPLACE pairs, applied in order. To insert code, SEARCH one anchor line "
    "and REPLACE it with the anchor plus the new lines. To delete code, leave the replacement empty.\n\n"
    "Rules: forward-slash paths relative to the project root. Touch only files you are allowed to. Prefer EDIT for "
    "existing files. Write complete, working code that matches the project's style, imports and interfaces: no "
    "placeholders, no '...' elisions, no TODO stubs. Never output the text " + PLACEHOLDER + "; read secrets from "
    "environment variables instead."
)

REVIEW_SYSTEM_PROMPT = "You are a senior code reviewer. Be terse and precise. Report only real defects."
ASK_SYSTEM_PROMPT = ("You are a senior engineer answering questions about a codebase. Be concise and concrete; "
                     "cite file:line. Plain text, no preamble.")

_BLOCK_RE = re.compile(r"^=== (FILE|EDIT): ?(.+?) ===[ \t]*\r?\n(.*?)\r?\n=== END (?:FILE|EDIT) ===[ \t]*$", re.S | re.M)
_ELISION_RES = [
    re.compile(r"^\s*(?:#|//|/\*+|<!--|--|\*)\s*(?:\.{3}|\[\.\.\.\]|\u2026)\s*.{0,40}?(?:rest|remaining|existing|unchanged|same|previous|code|more|other)", re.I | re.M),
    re.compile(r"^\s*(?:#|//|/\*+|<!--)\s*(?:rest of (?:the )?(?:code|file|function|class|implementation)|existing code (?:here|remains|unchanged)|remaining code|same as (?:before|above))", re.I | re.M),
]


class EditError(Exception):
    pass


def parse_blocks(text):
    """-> (blocks [(kind, path, body)], problems)"""
    problems = []
    starts = len(re.findall(r"^=== (?:FILE|EDIT):", text, re.M))
    blocks = [(k, _norm_rel(p), b) for k, p, b in _BLOCK_RE.findall(text)]
    if starts > len(blocks):
        problems.append("a block is missing its '=== END FILE ===' / '=== END EDIT ===' line (output cut off or malformed)")
    return blocks, problems


def _strip_fences(body):
    lines = body.strip("\n").split("\n")
    if len(lines) >= 2 and lines[0].lstrip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return body.strip("\n")


def parse_edit_pairs(body):
    pairs, errs = [], []
    lines = body.split("\n")
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip().startswith("<<<<<<< SEARCH"):
            i += 1
            s = []
            while i < n and lines[i].rstrip() != "=======":
                s.append(lines[i])
                i += 1
            if i >= n:
                errs.append("a SEARCH block has no '=======' line")
                break
            i += 1
            r = []
            while i < n and not lines[i].strip().startswith(">>>>>>> REPLACE"):
                r.append(lines[i])
                i += 1
            if i >= n:
                errs.append("a replacement has no '>>>>>>> REPLACE' line")
                break
            i += 1
            pairs.append(("\n".join(s), "\n".join(r)))
        else:
            i += 1
    if not pairs and not errs:
        errs.append("EDIT block contains no SEARCH/REPLACE pair")
    return pairs, errs


def _lead(line):
    return line[:len(line) - len(line.lstrip())]


def apply_edit(text, search, replace):
    """Applies one SEARCH/REPLACE to text. Returns (new_text, tier). Raises EditError."""
    if not search.strip():
        raise EditError("SEARCH text is empty (use a FILE block for new files)")

    def _splice(a, b, rep=None):
        rep = replace if rep is None else rep
        if rep == "" and (a == 0 or text[a - 1] == "\n") and (b == len(text) or text[b] == "\n"):
            if b < len(text):
                b += 1
            elif a > 0:
                a -= 1
        return text[:a] + rep + text[b:]

    n = text.count(search)
    if n == 1:
        a = text.index(search)
        return _splice(a, a + len(search)), "exact"
    if n > 1:
        raise EditError(f"SEARCH text matches {n} places; include more surrounding lines to make it unique")
    s2 = search.strip("\n")
    if s2 and s2 != search:
        n = text.count(s2)
        if n == 1:
            a = text.index(s2)
            return _splice(a, a + len(s2), replace.strip("\n")), "exact"
        if n > 1:
            raise EditError(f"SEARCH text matches {n} places; include more surrounding lines to make it unique")
    # tier 2: line-based, ignoring indentation / trailing spaces / blank lines
    t_lines = text.split("\n")
    s_raw = [l for l in search.strip("\n").split("\n")]
    s_norm = [l.strip() for l in s_raw if l.strip()]
    if s_norm:
        idx = [(i, l.strip()) for i, l in enumerate(t_lines) if l.strip()]
        norm_only = [x[1] for x in idx]
        hits = [j for j in range(len(norm_only) - len(s_norm) + 1) if norm_only[j:j + len(s_norm)] == s_norm]
        if len(hits) == 1:
            j = hits[0]
            a, b = idx[j][0], idx[j + len(s_norm) - 1][0]
            file_indent = _lead(t_lines[a])
            first_raw = next(l for l in s_raw if l.strip())
            search_indent = _lead(first_raw)
            new_lines = []
            if replace != "":
                for l in replace.strip("\n").split("\n"):
                    if not l.strip():
                        new_lines.append("")
                    elif file_indent != search_indent and l.startswith(search_indent):
                        new_lines.append(file_indent + l[len(search_indent):])
                    else:
                        new_lines.append(l)
            t_lines[a:b + 1] = new_lines
            return "\n".join(t_lines), "fuzzy"
        if len(hits) > 1:
            raise EditError(f"SEARCH text (ignoring indentation) matches {len(hits)} places; include more context")
    # tier 3: redacted-secret placeholder acts as a wildcard
    if PLACEHOLDER in search:
        parts = [re.escape(p) for p in search.split(PLACEHOLDER)]
        rx = re.compile(r"[^\s\"'`]+".join(parts))
        ms = list(rx.finditer(text))
        if len(ms) == 1:
            m = ms[0]
            return _splice(m.start(), m.end()), "wildcard"
        if len(ms) > 1:
            raise EditError("SEARCH text containing a redacted secret matches several places")
    raise EditError("SEARCH text not found in the file (it must be copied exactly from the current content)")


class PlanFile:
    __slots__ = ("rel", "before_bytes", "after_text", "eol", "bom", "new")

    def __init__(self, rel, before_bytes, after_text, eol, bom, new):
        self.rel, self.before_bytes, self.after_text = rel, before_bytes, after_text
        self.eol, self.bom, self.new = eol, bom, new

    @property
    def after_bytes(self):
        return encode_text(self.after_text, self.eol, self.bom)

    @property
    def before_text(self):
        if self.before_bytes is None:
            return ""
        got = None
        try:
            t = self.before_bytes.decode("utf-8-sig")
            got = t.replace("\r\n", "\n")
        except UnicodeDecodeError:
            got = ""
        return got


class Plan:
    def __init__(self):
        self.files = {}      # rel -> PlanFile (ordered)
        self.notes = []

    def merge(self, other):
        for rel, pf in other.files.items():
            if rel in self.files:
                prev = self.files[rel]
                pf.before_bytes, pf.new = prev.before_bytes, prev.new
            self.files[rel] = pf
        self.notes += other.notes
        return self


def _diff_stats(a_text, b_text):
    a, b = a_text.split("\n"), b_text.split("\n")
    if len(a) > 4000 and len(b) > 4000:
        sa, sb = set(a), set(b)
        return len([x for x in b if x not in sa]), len([x for x in a if x not in sb])
    add = rem = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            rem += i2 - i1
        if tag in ("replace", "insert"):
            add += j2 - j1
    return add, rem


def _validate_text(rel, before_text, after_text):
    """Deterministic quality gates on the changed content. Returns list of problems."""
    errs = []
    if PLACEHOLDER in after_text:
        errs.append(f"{rel}: contains the redacted-secret placeholder; use environment variables instead")
    if not after_text.strip():
        errs.append(f"{rel}: file would be empty")
        return errs
    lang = lang_of(rel)
    ext = os.path.splitext(rel)[1].lower()
    if lang == "python":
        def _ok(src):
            try:
                compile(src, rel, "exec")
                return None
            except SyntaxError as e:
                return f"line {e.lineno}: {e.msg}"
            except ValueError as e:
                return str(e)
        after_err = _ok(after_text)
        if after_err and (before_text == "" or _ok(before_text) is None):
            errs.append(f"{rel}: python syntax error at {after_err}")
    elif ext == ".json":
        def _jok(src):
            try:
                json.loads(src)
                return None
            except ValueError as e:
                return str(e)
        after_err = _jok(after_text)
        if after_err and (before_text == "" or _jok(before_text) is None):
            errs.append(f"{rel}: invalid JSON ({after_err})")
    return errs


def _elision_in(text):
    return any(rx.search(text) for rx in _ELISION_RES)


def build_plan(proj, answer, allowed, strict, overlay=None, single_target=None):
    """Turns a worker answer into a Plan. `allowed` = set of rel paths the worker may touch (existing files it
    saw + explicit new files); when not strict, any NEW file inside the project is also allowed.
    Returns (plan, problems)."""
    plan, problems = Plan(), []
    blocks, bp = parse_blocks(answer)
    problems += bp
    if not blocks and not bp and single_target:
        fences = re.findall(r"```[^\n]*\n(.*?)```", answer, re.S)
        if len(fences) == 1:
            blocks = [("FILE", single_target, fences[0])]
    if not blocks and not problems:
        problems.append("no '=== FILE:' or '=== EDIT:' blocks found in your answer")
        return plan, problems
    state = {}   # rel -> (text, eol, bom, before_bytes, new)

    def _load(rel):
        if rel in state:
            return state[rel]
        if overlay is not None and rel in overlay.files:
            pf = overlay.files[rel]
            state[rel] = (pf.after_text, pf.eol, pf.bom, pf.before_bytes, pf.new)
            return state[rel]
        p = proj.resolve(rel)
        if p.is_file():
            got = read_text_file(p)
            if got is None:
                return None
            state[rel] = (got[0], got[1], got[2], got[3], False)
        else:
            state[rel] = None
        return state[rel]

    for kind, rel, body in blocks:
        rel = proj.canon(rel)
        try:
            proj.resolve(rel)
        except ProjectError as e:
            problems.append(str(e))
            continue
        why = proj.write_block_reason(rel)
        if why:
            problems.append(f"{rel}: not allowed ({why})")
            continue
        existing = rel in proj.files or (overlay is not None and rel in overlay.files)
        if strict and rel not in allowed:
            problems.append(f"{rel}: not one of the files you may change ({', '.join(sorted(allowed)[:8])})")
            continue
        if not strict and existing and rel not in allowed:
            problems.append(f"{rel}: existing file you were not given; only edit the files shown as TARGET")
            continue
        cur = _load(rel)
        if kind == "FILE":
            content = _strip_fences(body)
            if _elision_in(content):
                problems.append(f"{rel}: contains an elision placeholder like '... rest of code'; write the complete file")
                continue
            if cur is not None:
                old_lines = cur[0].count("\n") + 1
                new_lines = content.count("\n") + 1
                if old_lines > 150:
                    problems.append(f"{rel}: existing file has {old_lines} lines; use EDIT blocks instead of rewriting it")
                    continue
                if old_lines > 40 and new_lines < old_lines * 0.5:
                    problems.append(f"{rel}: rewrite looks truncated ({new_lines} lines vs {old_lines}); use EDIT blocks")
                    continue
                state[rel] = (content.rstrip("\n") + "\n", cur[1], cur[2], cur[3], cur[4])
            else:
                state[rel] = (content.rstrip("\n") + "\n", "\n", False, None, True)
        else:
            if cur is None:
                problems.append(f"{rel}: EDIT used on a file that does not exist; use a FILE block to create it")
                continue
            pairs, errs = parse_edit_pairs(body)
            problems += [f"{rel}: {e}" for e in errs]
            text, fuzzy = cur[0], 0
            for k, (s, r) in enumerate(pairs, 1):
                if PLACEHOLDER in r:
                    problems.append(f"{rel}: edit {k} writes the redacted-secret placeholder; use os.environ / env vars")
                    continue
                if _elision_in(r):
                    problems.append(f"{rel}: edit {k} contains an elision placeholder; write the real code")
                    continue
                try:
                    text, tier = apply_edit(text, s, r)
                    if tier != "exact":
                        fuzzy += 1
                except EditError as e:
                    problems.append(f"{rel}: edit {k}: {e}")
            if fuzzy:
                plan.notes.append(f"{fuzzy} edit(s) in {rel} matched ignoring whitespace")
            state[rel] = (text, cur[1], cur[2], cur[3], cur[4])
    for rel, st in state.items():
        if st is None:
            continue
        text, eol, bom, before_bytes, is_new = st
        before_text = ""
        if before_bytes is not None:
            try:
                before_text = before_bytes.decode("utf-8-sig").replace("\r\n", "\n")
            except UnicodeDecodeError:
                before_text = ""
        if overlay is not None and rel in overlay.files:
            ov = overlay.files[rel]
            before_text = ov.before_text
        if not is_new and text == before_text:
            continue
        problems += _validate_text(rel, before_text, text)
        plan.files[rel] = PlanFile(rel, before_bytes, text, eol, bom, is_new)
    return plan, problems


# ===========================================================================
# Change sets: backup, apply, diff, undo
# ===========================================================================
class ChangeConflict(Exception):
    pass


def _next_change_id(proj):
    d = proj.data / "changes"
    n = 0
    if d.is_dir():
        for x in d.iterdir():
            m = re.match(r"c(\d+)$", x.name)
            if m:
                n = max(n, int(m.group(1)))
    return f"c{n + 1:03d}"


def _write_atomic_bytes(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".aiw_tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, p)


def _unified(rel, before_text, after_text, new):
    a = [] if new else before_text.split("\n")
    b = after_text.split("\n")
    if len(a) > 6000 or len(b) > 6000:
        return f"--- {rel}\n(diff skipped: file too large)\n"
    return "".join(l if l.endswith("\n") else l + "\n" for l in difflib.unified_diff(
        a, b, fromfile="/dev/null" if new else f"a/{rel}", tofile=f"b/{rel}", lineterm="", n=2))


def _load_meta(proj, cid):
    try:
        return json.loads((proj.data / "changes" / cid / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _save_meta(proj, cid, meta):
    (proj.data / "changes" / cid / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def commit_plan(proj, plan, desc, apply_now=True):
    """Stores the plan as a change set (with backups). Applies it unless apply_now is False.
    Returns the meta dict. Raises ChangeConflict when files changed since they were read."""
    with proj.lock:
        for rel, pf in plan.files.items():
            p = proj.resolve(rel)
            cur = p.read_bytes() if p.is_file() else None
            if sha(cur) != sha(pf.before_bytes):
                raise ChangeConflict(f"{rel} was changed on disk while the worker was running; retry the task")
        cid = _next_change_id(proj)
        base = proj.data / "changes" / cid
        meta = {"id": cid, "desc": desc[:120], "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "applied" if apply_now else "proposed", "files": {}}
        diff_parts = []
        for rel, pf in plan.files.items():
            ab = pf.after_bytes
            if pf.before_bytes is not None:
                _write_atomic_bytes(base / "before" / rel, pf.before_bytes)
            _write_atomic_bytes(base / "after" / rel, ab)
            add, rem = _diff_stats(pf.before_text, pf.after_text) if not pf.new else (pf.after_text.count("\n"), 0)
            meta["files"][rel] = {"before_sha": sha(pf.before_bytes), "after_sha": sha(ab), "new": pf.new,
                                  "add": add, "rem": rem}
            diff_parts.append(_unified(rel, pf.before_text, pf.after_text, pf.new))
        (base / "diff.patch").write_text("".join(diff_parts), encoding="utf-8")
        _save_meta(proj, cid, meta)
        if apply_now:
            _apply_files(proj, plan, meta)
        _prune_changes(proj)
        return meta


def _apply_files(proj, plan, meta):
    done = []
    try:
        for rel, pf in plan.files.items():
            _write_atomic_bytes(proj.resolve(rel), pf.after_bytes)
            done.append(rel)
    except OSError as e:
        for rel in done:  # roll back
            pf = plan.files[rel]
            p = proj.resolve(rel)
            try:
                if pf.before_bytes is None:
                    p.unlink()
                else:
                    _write_atomic_bytes(p, pf.before_bytes)
            except OSError:
                pass
        raise ProjectError(f"could not write files ({e}); nothing was changed")
    proj.refresh()


def _prune_changes(proj, keep=None):
    keep = keep or cfg_int("keep_changes", "AIW_KEEP_CHANGES", 60)
    d = proj.data / "changes"
    if not d.is_dir():
        return
    ids = sorted(x.name for x in d.iterdir() if re.match(r"c\d+$", x.name))
    for cid in ids[:-keep] if len(ids) > keep else []:
        meta = _load_meta(proj, cid) or {}
        if meta.get("status") != "proposed":
            shutil.rmtree(d / cid, ignore_errors=True)


def apply_change(proj, cid):
    with proj.lock:
        meta = _load_meta(proj, cid)
        if not meta:
            raise ProjectError(f"unknown change id {cid}")
        if meta["status"] == "applied":
            raise ProjectError(f"{cid} is already applied")
        if meta["status"] == "undone":
            raise ProjectError(f"{cid} was undone; re-run the task instead")
        base = proj.data / "changes" / cid
        plan = Plan()
        for rel, f in meta["files"].items():
            p = proj.resolve(rel)
            cur = p.read_bytes() if p.is_file() else None
            if sha(cur) != f["before_sha"]:
                raise ChangeConflict(f"{rel} changed since the proposal was made; re-run the task")
            after = (base / "after" / rel).read_bytes()
            got_before = (base / "before" / rel).read_bytes() if (base / "before" / rel).exists() else None
            pf = _RawPlanFile(rel, got_before, "", "\n", False, f["new"])
            pf.raw_after = after
            plan.files[rel] = pf
        _apply_files(proj, plan, meta)
        meta["status"] = "applied"
        _save_meta(proj, cid, meta)
        return meta


class _RawPlanFile(PlanFile):
    __slots__ = ("raw_after",)

    @property
    def after_bytes(self):
        return self.raw_after


def undo_change(proj, cid="", force=False):
    with proj.lock:
        if not cid:
            d = proj.data / "changes"
            ids = sorted((x.name for x in d.iterdir() if re.match(r"c\d+$", x.name)), reverse=True) if d.is_dir() else []
            for i in ids:
                m = _load_meta(proj, i)
                if m and m["status"] == "applied":
                    cid = i
                    break
            if not cid:
                raise ProjectError("nothing to undo")
        meta = _load_meta(proj, cid)
        if not meta:
            raise ProjectError(f"unknown change id {cid}")
        if meta["status"] != "applied":
            raise ProjectError(f"{cid} is not applied (status: {meta['status']})")
        base = proj.data / "changes" / cid
        restored, conflicts = [], []
        for rel, f in meta["files"].items():
            p = proj.resolve(rel)
            cur = p.read_bytes() if p.is_file() else None
            if sha(cur) != f["after_sha"] and not force:
                conflicts.append(rel)
                continue
            if f["new"]:
                try:
                    if p.exists():
                        p.unlink()
                    q = p.parent
                    while q != proj.root and _within(q, proj.root):
                        try:
                            q.rmdir()
                        except OSError:
                            break
                        q = q.parent
                except OSError:
                    conflicts.append(rel)
                    continue
            else:
                _write_atomic_bytes(p, (base / "before" / rel).read_bytes())
            restored.append(rel)
        if restored and not conflicts:
            meta["status"] = "undone"
            _save_meta(proj, cid, meta)
        proj.refresh()
        return cid, restored, conflicts


def describe_meta(meta):
    parts = []
    for rel, f in meta["files"].items():
        if f["new"]:
            parts.append(f"A {rel} +{f['add']}")
        else:
            parts.append(f"M {rel} +{f['add']} -{f['rem']}")
    return parts


# ===========================================================================
# Prompt building and the work pipeline
# ===========================================================================
def _short(rels, n=6):
    return ", ".join(rels[:n]) + (f" (+{len(rels) - n} more)" if len(rels) > n else "")


def _build_work_prompt(proj, task, targets, new_targets, context, worker, extra_note="", overlay=None):
    limit = max_prompt_chars(worker)
    budget = limit - 5000
    if budget < 4000:
        raise ProjectError(f"the models available allow only {limit} characters per request; raise "
                           "AIW_GROK_MAX_CHARS / AIW_GEMINI_MAX_CHARS or add a Gemini key")
    tgt_text, tgt_packed, tgt_skipped = "", [], []
    if overlay is not None:
        parts = []
        for rel in targets:
            pf = overlay.files.get(rel)
            if pf is not None:
                t, _n = redact(pf.after_text)
                parts.append(f"=== TARGET: {rel} ({t.count(chr(10)) + 1} lines) ===\n{t}\n=== END TARGET ===")
                tgt_packed.append(rel)
            else:
                t, p2, s2 = proj.pack([rel], budget, label="TARGET")
                if t:
                    parts.append(t)
                    tgt_packed += p2
                tgt_skipped += s2
        tgt_text = "\n\n".join(parts)
    else:
        tgt_text, tgt_packed, tgt_skipped = proj.pack(targets, budget, label="TARGET")
    if tgt_skipped:
        raise ProjectError("target files are too large for the models available (" + _short(tgt_skipped) +
                           f"; limit {limit} chars per request). Split the task by file/function, or add a Gemini key.")
    used = len(tgt_text)
    brief = proj.brief()
    map_budget = min(10000, max(1500, int((budget - used) * 0.15)))
    pmap = proj.map_text(map_budget)
    used += len(brief) + len(pmap)
    ctx_text, ctx_packed, ctx_skipped = proj.pack(
        [c for c in context if c not in targets], max(0, budget - used - 2000), label="CONTEXT")
    allowed_line = ", ".join(sorted(set(targets) | set(new_targets))) or "(none)"
    parts = []
    if brief:
        parts.append("PROJECT BRIEF:\n" + brief.strip())
    parts.append("PROJECT MAP (path, lines: symbols):\n" + pmap)
    if ctx_text:
        parts.append("CONTEXT FILES (read-only, do not change):\n" + ctx_text)
    if tgt_text:
        parts.append("TARGET FILES (current content; you may EDIT these):\n" + tgt_text)
    if new_targets:
        parts.append("NEW FILES TO CREATE (use FILE blocks): " + ", ".join(new_targets))
    parts.append("YOU MAY CHANGE ONLY: " + allowed_line + (" (plus new files you need)" if not new_targets and not targets else ""))
    if extra_note:
        parts.append(extra_note)
    parts.append("TASK:\n" + task.strip())
    return "\n\n".join(parts), tgt_packed, ctx_packed, ctx_skipped


def _extract_review_ids(task, proj):
    ids = sorted(set(int(x) for x in re.findall(r"\bF(\d{1,3})\b", task)))
    if not ids:
        return "", []
    data = _read_json(proj.data / "reports" / "last_review.json", None)
    if not data:
        return "", []
    byid = {f["id"]: f for f in data.get("findings", [])}
    lines, files = [], []
    for i in ids:
        f = byid.get(f"F{i}")
        if f:
            lines.append(f"F{i} [{f['sev']}] {f['file']}:{f['line']} - {f['problem']} - suggested fix: {f['fix']}")
            if f["file"] not in files:
                files.append(f["file"])
    if not lines:
        return "", []
    return "REVIEW FINDINGS TO FIX (verify each against the code first):\n" + "\n".join(lines), files


def _worker_check(worker):
    w = (worker or "fast").strip().lower()
    if w not in CHAINS:
        raise ProjectError("worker must be fast, deep or creative")
    return w


def _work_core(proj, task, files=None, context=None, worker="fast", review=False, apply=True, hint_files=None):
    """The edit pipeline. Returns (ok, text, change_id)."""
    worker = _worker_check(worker)
    if not task.strip():
        return False, "ERROR: empty task", None
    finding_note, finding_files = _extract_review_ids(task, proj)
    explicit = files is not None and len(files) > 0
    found, unmatched = proj.expand(files) if explicit else ([], [])
    targets, new_targets = list(found), []
    strict = explicit
    for pat in unmatched:
        if any(c in pat for c in "*?"):
            return False, f"ERROR: pattern '{pat}' matches no files; give exact paths for new files", None
        rel = proj.canon(pat)
        try:
            proj.resolve(rel)
        except ProjectError as e:
            return False, f"ERROR: {e}", None
        why = proj.write_block_reason(rel)
        if why:
            return False, f"ERROR: {rel} cannot be created ({why})", None
        new_targets.append(rel)
    if explicit:
        for t in targets:
            why = proj.write_block_reason(t)
            if why:
                return False, f"ERROR: {t} cannot be changed ({why})", None
    ctx, ctx_unmatched = proj.expand(context or [])
    if ctx_unmatched:
        return False, "ERROR: context not found: " + ", ".join(ctx_unmatched), None
    auto_note = ""
    if not explicit:
        base = list(finding_files) or list(hint_files or [])
        picks = [p for p in dict.fromkeys(base + proj.select(task, k=4)) if proj.is_text(p)]
        picks = [p for p in picks if not proj.write_block_reason(p)]
        targets = picks[:5]
        if not targets:
            return False, "ERROR: could not find relevant files for this task; pass files=[...]", None
        auto_note = "auto-selected: " + _short(targets)
    neighbor = proj.neighbors(targets) if not ctx else []
    context_all = list(dict.fromkeys(ctx + neighbor))
    allowed = set(targets) | set(new_targets)
    try:
        prompt0, seen, ctx_seen, ctx_skipped = _build_work_prompt(
            proj, task, targets, new_targets, context_all, worker, extra_note=finding_note)
    except ProjectError as e:
        return False, f"ERROR: {e}", None
    seen_set = set(seen)
    allowed = (allowed & (seen_set | set(new_targets))) if strict else (seen_set | set(new_targets))
    single = new_targets[0] if (len(new_targets) == 1 and not targets) else None
    label = "work: " + task.strip()[:30]
    attempt_prompt, plan, problems, res = prompt0, Plan(), [], None
    for attempt in range(max_fixes() + 1):
        res = run_worker(worker, attempt_prompt, system=EDIT_SYSTEM_PROMPT, label=label)
        if not isinstance(res, Result):
            return False, str(res), None
        plan, problems = build_plan(proj, str(res), allowed, strict, single_target=single)
        if res.truncated:
            problems.insert(0, "output was cut off at the model token limit; return fewer/shorter changes (use EDIT blocks)")
        if not problems and not plan.files:
            problems = ["your answer produced no change to any file"]
        if not problems:
            break
        note = ("\n\nYOUR PREVIOUS ANSWER WAS REJECTED:\n" + "\n".join(f"- {p}" for p in problems[:8]) +
                "\nReturn the corrected change blocks (all of them) in the required format.")
        if not res.truncated and len(res) < 20000:
            note = "\n\nYOUR PREVIOUS ANSWER:\n" + str(res) + note
        attempt_prompt = prompt0 + note
    if problems:
        rej = proj.data / "rejected"
        rej.mkdir(parents=True, exist_ok=True)
        rp = rej / (time.strftime("%Y%m%d-%H%M%S") + ".txt")
        try:
            rp.write_text(str(res), encoding="utf-8")
        except OSError:
            pass
        msg = [f"FAILED after {max_fixes() + 1} attempt(s) via {getattr(res, 'handled_by', '?')}. Nothing was changed."]
        msg += [f"- {p}" for p in problems[:5]]
        msg.append(f"Last worker answer saved: {rp}")
        msg.append("Retry with a smaller task, explicit files=[...], or worker='deep'.")
        return False, "\n".join(msg), None
    notes = list(plan.notes)
    if review:
        rtxt, _fp = _review_text_for_plan(proj, plan)
        if rtxt is None:
            notes.append("review: unavailable (all reviewers failed)")
        elif rtxt == "":
            notes.append("review: no issues")
        else:
            fixed = False
            fix_prompt, _s, _c, _k = _build_work_prompt(
                proj, task, list(plan.files.keys()) if strict is False else [t for t in plan.files], [],
                context_all, worker, overlay=plan,
                extra_note="A code review of the CURRENT target files (shown above) found these issues; fix them with "
                           "EDIT blocks:\n" + rtxt[:3000])
            res2 = run_worker(worker, fix_prompt, system=EDIT_SYSTEM_PROMPT, label=label + " (fix)")
            if isinstance(res2, Result) and not res2.truncated:
                allowed2 = set(plan.files.keys())
                plan2, probs2 = build_plan(proj, str(res2), allowed2, True, overlay=plan)
                if not probs2 and plan2.files:
                    plan.merge(plan2)
                    fixed = True
            n = len([l for l in rtxt.splitlines() if l.strip()])
            notes.append(f"review: {n} issue(s) found and auto-fixed" if fixed else
                         f"review: {n} issue(s) found, auto-fix failed -> " + rtxt[:400])
    try:
        meta = commit_plan(proj, plan, task, apply_now=apply)
    except (ChangeConflict, ProjectError) as e:
        return False, f"ERROR: {e}", None
    head = ("OK " if apply else "PROPOSED ") + f"{meta['id']} via {res.handled_by} | read {len(seen)} file(s)"
    if ctx_seen:
        head += f" + {len(ctx_seen)} context"
    lines = [head, " | ".join(describe_meta(meta))]
    if auto_note:
        lines.append(auto_note)
    if ctx_skipped:
        lines.append("context left out (too big): " + _short(ctx_skipped, 4))
    py = [r for r in plan.files if lang_of(r) == "python"]
    chk = []
    if py:
        chk.append("python syntax OK")
    if chk:
        lines.append("checks: " + ", ".join(chk))
    lines += notes
    if apply:
        lines.append(f"undo: changes(action='undo', id='{meta['id']}')")
    else:
        lines.append(f"diff: {proj.data / 'changes' / meta['id'] / 'diff.patch'} | apply: changes(action='apply', id='{meta['id']}')")
    return True, "\n".join(lines), meta["id"]


# ===========================================================================
# Review (map-reduce) and ask
# ===========================================================================
_FINDING_RE = re.compile(r"^\W*(?:F?\d+[.)]\s*)?(HIGH|MEDIUM|MED|LOW)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*$", re.I)
REVIEW_FORMAT = (
    "Return findings, one per line, exactly:\n"
    "SEV | path:line | problem (max 20 words) | fix (max 20 words)\n"
    "SEV is HIGH (security hole, data loss, crash, wrong results), MED (real bug or fragile logic) or LOW. "
    "At most 12 findings, most severe first. Only real, concrete defects you can point to in the code shown: "
    "bugs, security flaws, broken imports/interfaces between files, missing error handling that causes failures. "
    "No style remarks. If nothing is wrong reply exactly: NONE")


def parse_findings(text):
    out = []
    for line in str(text).splitlines():
        m = _FINDING_RE.match(line.strip())
        if not m:
            continue
        sev = m.group(1).upper()
        sev = "MED" if sev.startswith("MED") else sev
        loc = m.group(2).strip().strip("`")
        mm = re.match(r"(.+?)(?::(\d+))?(?:[-,]\d+)?$", loc)
        path = _norm_rel(mm.group(1)) if mm else _norm_rel(loc)
        line_no = int(mm.group(2)) if mm and mm.group(2) else 0
        out.append({"sev": sev, "file": path, "line": line_no, "problem": m.group(3).strip(), "fix": m.group(4).strip()})
    return out


def _review_text_for_plan(proj, plan):
    """Reviews the new content of a plan. Returns ('' | issues | None, findings)"""
    parts = []
    for rel, pf in plan.files.items():
        t, _n = redact(pf.after_text)
        lines = t.split("\n")
        parts.append(f"=== FILE: {rel} ===\n" + "\n".join(f"{i:>4}| {l}" for i, l in enumerate(lines, 1)) + "\n=== END FILE ===")
    body = "\n\n".join(parts)
    limit = max_prompt_chars("deep") - 6000
    if len(body) > limit:
        body = body[:limit] + "\n[...truncated...]"
    prompt = (("PROJECT BRIEF:\n" + proj.brief() + "\n\n") if proj.brief() else "") + REVIEW_FORMAT + "\n\n" + body
    res = run_worker("deep", prompt, system=REVIEW_SYSTEM_PROMPT, label="review")
    if not isinstance(res, Result):
        return None, []
    fnd = parse_findings(res)
    fnd = [f for f in fnd if f["sev"] in ("HIGH", "MED")]
    if not fnd:
        return "", []
    return "\n".join(f"- {f['file']}:{f['line']} - {f['problem']} - {f['fix']}" for f in fnd[:8]), fnd


def _risk(proj, rel):
    low = rel.lower()
    s = sum(3 for w in RISK_WORDS if w in low)
    e = proj.files[rel]
    s += sum(1 for sy in e.get("sym", []) for w in ("login", "auth", "token", "password", "query", "upload") if w in sy.lower())
    if TEST_RE.search(rel):
        s -= 3
    return s


def _reviewable(proj, scope):
    if scope:
        found, _u = proj.expand(scope)
        rels = [r for r in found if proj.files[r].get("lang") in CODE_LANGS or r.endswith((".json", ".yml", ".yaml", ".toml", ".xml", ".gradle"))]
    else:
        rels = [r for r in proj.text_files() if proj.files[r].get("lang") in CODE_LANGS]
    return [r for r in rels if proj.files[r]["size"] <= 200_000 and not proj.files[r].get("sens")]


def do_review(proj, focus="", scope=None, verify=True):
    rels = _reviewable(proj, scope)
    if not rels:
        return "ERROR: no reviewable source files found" + (" for that scope" if scope else "")
    chunk_limit = min(cfg_int("review_chunk_chars", "AIW_REVIEW_CHUNK_CHARS", 90_000), int(max_prompt_chars("deep") * 0.6))
    chunk_limit = max(chunk_limit, 4000)
    max_chunks = cfg_int("review_max_batches", "AIW_REVIEW_MAX_BATCHES", 12)
    rels.sort(key=lambda r: (-_risk(proj, r), r))
    blocks = []
    for r in rels:
        t = proj.read_for_worker(r, numbered=True)
        if t is None:
            continue
        b = f"=== FILE: {r} ===\n{t}\n=== END FILE ==="
        if len(b) > chunk_limit:
            b = b[:chunk_limit - 100] + "\n[...file truncated for review...]\n=== END FILE ==="
        blocks.append((r, b))
    chunks, cur, size, skipped = [], [], 0, []
    for r, b in blocks:
        if size + len(b) > chunk_limit and cur:
            chunks.append(cur)
            cur, size = [], 0
        if len(chunks) >= max_chunks:
            skipped.append(r)
            continue
        cur.append((r, b))
        size += len(b)
    if cur and len(chunks) < max_chunks:
        chunks.append(cur)
    elif cur:
        skipped += [r for r, _b in cur]
    pmap = proj.map_text(min(6000, max(1000, chunk_limit // 8)))
    head = (("PROJECT BRIEF:\n" + proj.brief() + "\n\n") if proj.brief() else "") + "PROJECT MAP:\n" + pmap
    fx = f"\nExtra focus: {focus}\n" if focus.strip() else "\n"
    results, handled = [], []

    def _one(i):
        body = "\n\n".join(b for _r, b in chunks[i])
        prompt = head + "\n\nFILES TO REVIEW (batch %d of %d):\n%s\n\n%s%s" % (i + 1, len(chunks), body, REVIEW_FORMAT, fx)
        res = run_worker("deep", prompt, system=REVIEW_SYSTEM_PROMPT, label=f"review {i + 1}/{len(chunks)}")
        return i, res

    with ThreadPoolExecutor(max_workers=3) as pool:
        outs = list(pool.map(_one, range(len(chunks))))
    failed = 0
    allf = []
    for i, res in sorted(outs, key=lambda x: x[0]):
        if not isinstance(res, Result):
            failed += 1
            continue
        handled.append(res.handled_by)
        allf += parse_findings(res)
    if failed == len(chunks):
        return "ERROR: every review batch failed:\n" + str(outs[0][1])[:500]
    uniq = {}
    for f in allf:
        f["file"] = proj.canon(f["file"])
        key = (f["file"], f["line"] // 4, re.sub(r"\W+", " ", f["problem"].lower())[:30])
        if key not in uniq:
            uniq[key] = f
    findings = [f for f in uniq.values() if f["file"] in proj.files]   # drop hallucinated paths
    order = {"HIGH": 0, "MED": 1, "LOW": 2}
    findings.sort(key=lambda f: (order[f["sev"]], f["file"], f["line"]))
    removed, unverified = 0, False
    if verify:
        cand = [f for f in findings if f["sev"] in ("HIGH", "MED")][:12]
        if cand:
            parts = []
            for k, f in enumerate(cand, 1):
                got = proj.read_for_worker(f["file"], numbered=True) if f["file"] in proj.files else None
                snippet = ""
                if got:
                    ls = got.split("\n")
                    a = max(0, f["line"] - 14)
                    snippet = "\n".join(ls[a:f["line"] + 14])
                parts.append(f"[V{k}] {f['file']}:{f['line']} [{f['sev']}] {f['problem']}\n{snippet}")
            vp = ("Verify each reported finding against the code shown. Reply one line per finding: "
                  "'Vk | CONFIRM | reason' or 'Vk | REJECT | reason'. Reject if the code does not actually have the problem "
                  "or the problem is only theoretical.\n\n" + "\n\n".join(parts))
            vres = run_worker("deep", vp, system=REVIEW_SYSTEM_PROMPT, label="verify findings")
            if isinstance(vres, Result):
                rej = set(int(m.group(1)) for m in re.finditer(r"^\W*V(\d+)\s*\|\s*REJECT", str(vres), re.M | re.I))
                drop = {id(cand[k - 1]) for k in rej if 1 <= k <= len(cand)}
                removed = len(drop)
                findings = [f for f in findings if id(f) not in drop]
            else:
                unverified = True
    for n, f in enumerate(findings, 1):
        f["id"] = f"F{n}"
    rdir = proj.data / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "last_review.json").write_text(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "focus": focus,
                                                        "findings": findings}, indent=1), encoding="utf-8")
    md = [f"# Review of {proj.name} ({time.strftime('%Y-%m-%d %H:%M')})", f"Focus: {focus or 'general'}", ""]
    for f in findings:
        md.append(f"- **{f['id']} {f['sev']}** `{f['file']}:{f['line']}` - {f['problem']}  \n  Fix: {f['fix']}")
    rp = rdir / ("review-" + time.strftime("%Y%m%d-%H%M%S") + ".md")
    rp.write_text("\n".join(md) + "\n", encoding="utf-8")
    cnt = {s: len([f for f in findings if f["sev"] == s]) for s in ("HIGH", "MED", "LOW")}
    via = handled[0] if handled else "?"
    out = [f"Review: {len(blocks)} file(s) in {len(chunks)} batch(es) via {via} | HIGH {cnt['HIGH']}, MED {cnt['MED']}, LOW {cnt['LOW']}"
           + (f" (verification dropped {removed} false positive(s))" if removed else "")
           + (" (unverified)" if unverified else "")]
    for f in findings[:10]:
        out.append(f"{f['id']} {f['sev']} {f['file']}:{f['line']} - {f['problem'][:110]} -> {f['fix'][:80]}")
    if len(findings) > 10:
        out.append(f"... {len(findings) - 10} more in the report")
    if skipped:
        out.append(f"Not reviewed (batch cap): {_short(skipped, 5)} - run review(scope=[...]) for them")
    if failed:
        out.append(f"{failed} batch(es) failed (rate limits?); re-run to cover them")
    out.append(f"Report: {rp}")
    if findings:
        out.append("Fix with: work(task='fix F1,F2') - findings are expanded server-side")
    return "\n".join(out)


def do_ask(proj, question, files=None, deep=False):
    worker = "deep" if deep else "fast"
    if files:
        rels, unmatched = proj.expand(files)
        if unmatched:
            return "ERROR: not found: " + ", ".join(unmatched)
    else:
        rels = proj.select(question, k=8)
        rels = list(dict.fromkeys(rels + proj.neighbors(rels[:3], limit=3)))
    if not rels:
        return "ERROR: no relevant files found; pass files=[...]"
    limit = max_prompt_chars(worker) - 6000
    pmap = proj.map_text(min(6000, max(1000, limit // 8)))
    brief = proj.brief()
    text, packed, skipped = proj.pack(rels, limit - len(pmap) - len(brief) - 500, label="FILE", numbered=True)
    prompt = (("PROJECT BRIEF:\n" + brief + "\n\n") if brief else "") + "PROJECT MAP:\n" + pmap + "\n\nFILES:\n" + text + \
             "\n\nQUESTION: " + question.strip() + "\n\nAnswer in at most 15 lines."
    res = run_worker(worker, prompt, system=ASK_SYSTEM_PROMPT, label="ask: " + question[:30])
    if not isinstance(res, Result):
        return str(res)
    out = str(res).strip()
    if len(out) > 3500:
        out = out[:3500] + "\n...(cut)"
    return out + f"\n[read: {_short(packed, 6)}" + (f"; too big to include: {_short(skipped, 3)}" if skipped else "") + "]"


# ===========================================================================
# run_check (user-approved commands only) + auto-fix loop
# ===========================================================================
def _project_python(proj):
    for cand in ("venv/Scripts/python.exe", ".venv/Scripts/python.exe", "venv/bin/python", ".venv/bin/python"):
        p = proj.root / cand
        if p.is_file():
            return str(p)
    return shutil.which("python") or shutil.which("python3") or sys.executable


def check_commands(proj):
    """Named commands that run_check may execute: user-configured + safe presets."""
    cmds = {}
    conf = (_cfg_data().get("checks") or {}).get(proj.name) or {}
    if isinstance(conf, dict):
        for k, v in conf.items():
            if isinstance(v, str) and v.strip():
                cmds[k] = v.strip()
    text = proj.text_files()
    if "tests" not in cmds and any(r.lower().rsplit("/", 1)[-1].startswith("test_") and r.endswith(".py") for r in text):
        cmds["tests"] = f"\"{_project_python(proj)}\" -m pytest -q -x"
    pj = [r for r in text if r.lower() == "package.json"]
    if pj:
        try:
            scripts = (json.loads((proj.read(pj[0]) or ("{}",))[0]).get("scripts") or {})
        except ValueError:
            scripts = {}
        for name in ("test", "build", "lint", "typecheck"):
            key = "tests" if name == "test" else name
            if name in scripts and key not in cmds:
                cmds[key] = f"npm run {name}"
    if "pom.xml" in [r.lower() for r in text] and "tests" not in cmds and shutil.which("mvn"):
        cmds["tests"] = "mvn -q test"
    return cmds


def _strip_secret_env():
    env = dict(os.environ)
    for k in list(env):
        if k.startswith(("GEMINI_", "XAI_", "GROK_", "AIW_")) or "API_KEY" in k:
            env.pop(k, None)
    return env


def _run_command(proj, cmdline, timeout):
    posix = os.name != "nt"
    args = shlex.split(cmdline, posix=posix)
    if not posix:
        args = [a[1:-1] if len(a) > 1 and a[0] == a[-1] == '"' else a for a in args]
    exe = shutil.which(args[0]) or args[0]
    args[0] = exe
    t0 = time.time()
    try:
        cp = subprocess.run(args, cwd=str(proj.root), capture_output=True, text=True, timeout=timeout,
                            env=_strip_secret_env(), encoding="utf-8", errors="replace")
        out = (cp.stdout or "") + (cp.stderr or "")
        return cp.returncode, out, time.time() - t0
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")) if e else ""
        return 124, out + f"\n[timed out after {timeout}s]", time.time() - t0
    except (OSError, ValueError) as e:
        return 127, f"could not run command: {e}", time.time() - t0


def _tail(out, lines=45, chars=3500):
    t = "\n".join(out.strip().splitlines()[-lines:])
    return t[-chars:]


def _files_from_output(proj, out):
    found = []
    for m in re.finditer(r"([A-Za-z0-9_./\\:-]+\.(?:py|js|jsx|ts|tsx|java|kt|go|rs|rb|php|cs|vue))(?::\d+|\", line \d+|\()", out):
        raw = m.group(1).replace("\\", "/")
        for rel in proj.files:
            if raw.endswith(rel) or rel.endswith(raw.lstrip("./")):
                if rel not in found:
                    found.append(rel)
                break
    return found[:5]


def do_run_check(proj, name="", fix=False):
    if not cfg_bool("allow_run", "AIW_ALLOW_RUN", False):
        return ("ERROR: running commands is switched off. To allow tests/builds, run 'python server.py --setup' and "
                "answer yes to 'allow running tests', or set AIW_ALLOW_RUN=1.")
    cmds = check_commands(proj)
    if not cmds:
        return ("ERROR: no check command found for this project. Add one: python server.py --set-check "
                f"\"{proj.name}\" tests \"<command>\"")
    if not name:
        name = "tests" if "tests" in cmds else next(iter(cmds))
    if name not in cmds:
        return f"ERROR: unknown check '{name}'. Available: {', '.join(cmds)}"
    cmdline = cmds[name]
    timeout = cfg_int("check_timeout", "AIW_CHECK_TIMEOUT", 240)
    rounds = cfg_int("fix_rounds", "AIW_FIX_ROUNDS", 2)
    changes_made = []
    for rnd in range(rounds + 1):
        code, out, secs = _run_command(proj, cmdline, timeout)
        if code == 0:
            msg = f"PASS {name} ({secs:.1f}s)"
            if changes_made:
                msg += f" after {len(changes_made)} auto-fix round(s): " + ", ".join(changes_made)
            return msg
        if not fix or rnd == rounds:
            return f"FAIL {name} exit={code} ({secs:.1f}s)" + (f" after {len(changes_made)} fix round(s): " + ", ".join(changes_made) if changes_made else "") + "\n" + _tail(out)
        task = (f"The command `{cmdline}` failed (exit {code}). Output tail:\n{_tail(out)}\n\n"
                "Find the root cause in the source code and fix it. Do not delete or weaken tests unless a test is clearly wrong.")
        hints = _files_from_output(proj, out)
        ok, text, cid = _work_core(proj, task, files=None, context=None, worker="deep", review=False, apply=True, hint_files=hints)
        if not ok:
            return f"FAIL {name} exit={code}; auto-fix round {rnd + 1} failed:\n{text[:700]}\n--- output ---\n{_tail(out, 15, 1200)}"
        changes_made.append(cid)
        proj.refresh()
    return "FAIL"


# ===========================================================================
# MCP tools
# ===========================================================================
def _wrap(fn):
    def inner(*a, **k):
        try:
            return fn(*a, **k)
        except ProjectError as e:
            return f"ERROR: {e}"
        except Exception as e:  # never crash the server
            log(f"internal error in {fn.__name__}: {e!r}")
            return f"ERROR: {fn.__name__} failed: {e}"
    inner.__name__ = fn.__name__
    inner.__doc__ = fn.__doc__
    inner.__wrapped__ = fn
    return inner


def _tool(fn):
    """Registers fn as an MCP tool with error wrapping; keeps the original signature for the schema."""
    import functools
    import inspect

    @functools.wraps(fn)
    def inner(*a, **k):
        try:
            return fn(*a, **k)
        except ProjectError as e:
            return f"ERROR: {e}"
        except Exception as e:
            log(f"internal error in {fn.__name__}: {e!r}")
            return f"ERROR: {fn.__name__} failed: {e}"
    inner.__signature__ = inspect.signature(fn)
    mcp.tool()(inner)
    return inner


@_tool
def open_project(project: str = "", new: str = "", brief: str = "") -> str:
    """Open a project. project=<full folder path or name> opens an existing local folder (must be in the allowed list); new=<name> creates a fresh project; brief=<=200 words of stack/conventions shared with every worker. No args: list allowed folders."""
    if project and new:
        raise ProjectError("give either project or new, not both")
    if not project and not new:
        if ACTIVE["project"] is not None and not brief:
            return list_projects_text() + f"\nCurrently open: {ACTIVE['project'].name}"
        if brief and ACTIVE["project"] is not None:
            ACTIVE["project"].set_brief(brief)
            return f"Brief saved for '{ACTIVE['project'].name}' ({len(brief)} chars)."
        return list_projects_text()
    if new:
        name = re.sub(r"[^A-Za-z0-9_. -]+", "_", new.strip()).strip(" .") or "project"
        root = workspace_dir() / name
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = find_project_dir(project)
    proj = Project(root)
    with _active_lock:
        ACTIVE["project"] = proj
    if brief.strip():
        proj.set_brief(brief)
    msg = proj.summary()
    if new:
        msg += "\n(new project folder created)"
    return msg


@_tool
def ask(question: str, files: list[str] | None = None, deep: bool = False) -> str:
    """Read-only. A worker reads the relevant files (auto-picked, or given paths/globs) and answers concisely. Use it to understand code instead of reading it yourself."""
    return do_ask(_active(), question, files, deep)


@_tool
def review(focus: str = "", scope: list[str] | None = None) -> str:
    """Audit the whole project (or scope globs) in parallel batches for bugs, security and logic problems. Returns ranked findings F1..Fn; fix them with work(task='fix F1,F2')."""
    return do_review(_active(), focus, scope)


@_tool
def work(task: str, files: list[str] | None = None, context: list[str] | None = None, worker: str = "fast",
         review: bool = False, apply: bool = True) -> str:
    """Create or change code; the worker reads the files itself and edits on disk. files = paths/globs it may edit or create (omit to auto-pick). context = extra read-only paths. worker = fast|deep|creative. review=True adds a review+auto-fix pass. apply=False only proposes a diff. task may cite review ids (F1,F2)."""
    ok, text, _cid = _work_core(_active(), task, files, context, worker, review, apply)
    return text


@_tool
def swarm(tasks: list[dict]) -> str:
    """Run up to 3 independent work tasks in parallel: [{task, files, context, worker, review}]. Every task needs explicit files and the file sets must not overlap."""
    proj = _active()
    if not isinstance(tasks, list) or not tasks:
        raise ProjectError("provide a non-empty list of tasks")
    tasks = tasks[:MAX_SWARM_TASKS]
    seen, norm = {}, []
    for i, t in enumerate(tasks, 1):
        if not isinstance(t, dict) or not t.get("task") or not t.get("files"):
            raise ProjectError(f"task {i}: needs 'task' and explicit 'files'")
        found, unmatched = proj.expand(t["files"])
        paths = set(found) | {proj.canon(u) for u in unmatched}
        for p in paths:
            if p in seen:
                raise ProjectError(f"tasks {seen[p]} and {i} both change {p}; split the files so they do not overlap")
            seen[p] = i
        norm.append(t)

    def _one(t):
        return _work_core(proj, str(t["task"]), t.get("files"), t.get("context"), str(t.get("worker", "fast")),
                          bool(t.get("review", False)), bool(t.get("apply", True)))

    with ThreadPoolExecutor(max_workers=MAX_SWARM_TASKS) as pool:
        futs = [pool.submit(_one, t) for t in norm]
        outs = []
        for f in futs:
            try:
                outs.append(f.result()[1])
            except Exception as e:
                outs.append(f"ERROR: {e}")
    return "\n\n".join(f"## Task {i} ({str(t.get('worker', 'fast'))})\n{o}" for i, (t, o) in enumerate(zip(norm, outs), 1))


@_tool
def changes(action: str = "list", id: str = "", force: bool = False) -> str:
    """action: list | diff (of id) | apply (a proposed change id) | undo (id, default latest; restores files from before the change)."""
    proj = _active()
    action = action.strip().lower()
    if action == "list":
        d = proj.data / "changes"
        ids = sorted((x.name for x in d.iterdir() if re.match(r"c\d+$", x.name)), reverse=True)[:8] if d.is_dir() else []
        if not ids:
            return "No changes yet."
        rows = []
        for cid in ids:
            m = _load_meta(proj, cid)
            if m:
                rows.append(f"{cid} {m['status']:<8} {m['ts']} {m['desc'][:50]} | {len(m['files'])} file(s): {_short(list(m['files']), 3)}")
        return "\n".join(rows)
    if action == "diff":
        if not id:
            raise ProjectError("give the change id")
        p = proj.data / "changes" / id / "diff.patch"
        if not p.is_file():
            raise ProjectError(f"unknown change id {id}")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        out = "\n".join(lines[:140])
        return out + (f"\n... ({len(lines) - 140} more lines) full diff: {p}" if len(lines) > 140 else f"\n(full diff: {p})")
    if action == "apply":
        if not id:
            raise ProjectError("give the proposed change id")
        try:
            meta = apply_change(proj, id)
        except ChangeConflict as e:
            return f"ERROR: {e}"
        return f"OK applied {id}: " + " | ".join(describe_meta(meta)) + f"\nundo: changes(action='undo', id='{id}')"
    if action == "undo":
        cid, restored, conflicts = undo_change(proj, id, force)
        msg = f"Undid {cid}: restored {_short(restored, 6) or 'nothing'}"
        if conflicts:
            msg += f"\nSkipped (edited since the change): {_short(conflicts, 6)} - pass force=True to overwrite them"
        return msg
    raise ProjectError("action must be list, diff, apply or undo")


@_tool
def run_check(name: str = "", fix: bool = False) -> str:
    """Run a user-approved project command (tests/build/lint; needs allow_run). fix=True lets workers auto-fix failures and re-run (up to 2 rounds). Returns PASS or the failure tail."""
    return do_run_check(_active(), name, fix)


@_tool
def view(path: str = "", start: int = 1, end: int = 0) -> str:
    """No path: project map. With a path: numbered lines (max 200). For debugging only; prefer ask()."""
    proj = _active()
    if not path.strip():
        return proj.map_text(7000)
    rel = proj.canon(path)
    if rel not in proj.files or not proj.is_text(rel):
        raise ProjectError(f"{rel} not found or not a readable text file")
    got = proj.read_for_worker(rel)
    lines = got.split("\n")
    start = max(1, int(start))
    end = int(end) if int(end) > 0 else start + TEXT_PREVIEW_CAP - 1
    end = min(end, start + TEXT_PREVIEW_CAP - 1, len(lines))
    return f"{rel} lines {start}-{end} of {len(lines)}\n" + "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))


@_tool
def patch(path: str, old: str, new: str) -> str:
    """Replace one snippet in a project file (tiny edits only; undoable). old must match once."""
    proj = _active()
    rel = proj.canon(path)
    why = proj.write_block_reason(rel)
    if why:
        raise ProjectError(f"{rel} cannot be changed ({why})")
    p = proj.resolve(rel)
    if not p.is_file():
        raise ProjectError(f"{rel} not found")
    got = read_text_file(p)
    if got is None:
        raise ProjectError(f"{rel} is not a text file")
    try:
        text, _tier = apply_edit(got[0], old, new)
    except EditError as e:
        raise ProjectError(str(e))
    errs = _validate_text(rel, got[0], text)
    if errs:
        return "REFUSED (file unchanged): " + "; ".join(errs)
    plan = Plan()
    plan.files[rel] = PlanFile(rel, got[3], text, got[1], got[2], False)
    meta = commit_plan(proj, plan, f"patch {rel}")
    return f"OK {meta['id']}: " + " | ".join(describe_meta(meta))


@_tool
def status() -> str:
    """Markdown table: which model handled each task, timings, fallbacks."""
    with _status_lock:
        rows = list(_status_log)[-30:]
    if not rows:
        return "No tasks executed yet."
    lines = ["| # | Worker | Task | Handled By | Key | Time (s) | Fallbacks | Status |",
             "|---|--------|------|------------|-----|----------|-----------|--------|"]
    for r in rows:
        lines.append(f"| {r['n']} | {r['worker']} | {r['task'].replace('|', '/')} | {r['handled_by']} | {r['key']} | "
                     f"{r['seconds']} | {r['fallbacks']} | {'OK' if r['ok'] else 'FAILED'} |")
    return "\n".join(lines)


# ===========================================================================
# CLI: setup wizard, check, project registry, Claude Desktop config
# ===========================================================================
_input = input
_getpass = getpass.getpass


def claude_config_paths():
    p = os.environ.get("AIW_CLAUDE_CONFIG")
    if p:
        return [Path(p)]
    if os.name == "nt":
        out = []
        if os.environ.get("APPDATA"):
            out.append(Path(os.environ["APPDATA"]) / "Claude" / "claude_desktop_config.json")
        if os.environ.get("LOCALAPPDATA"):
            out += sorted(Path(os.environ["LOCALAPPDATA"], "Packages").glob("Claude_*/LocalCache/Roaming/Claude/claude_desktop_config.json"))
        return out or [Path.home() / "AppData/Roaming/Claude/claude_desktop_config.json"]
    if sys.platform == "darwin":
        return [Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"]
    return [Path.home() / ".config/Claude/claude_desktop_config.json"]


def server_python():
    for cand in ("venv/Scripts/python.exe", "venv/bin/python"):
        p = HERE / cand
        if p.is_file():
            return str(p)
    return sys.executable


def install_claude_config():
    paths = claude_config_paths()
    existing = [p for p in paths if p.exists()]
    target = existing[0] if existing else paths[0]
    data = {}
    if target.exists():
        raw = target.read_text(encoding="utf-8-sig")
        if raw.strip():
            try:
                data = json.loads(raw)
            except ValueError as e:
                print(f"  Could not read {target}: it is not valid JSON ({e}).")
                print("  I did not touch it. Fix or delete that file, then run this again.")
                return False
        shutil.copy2(target, target.with_name(target.name + ".bak-aiw"))
    if not isinstance(data, dict):
        data = {}
    servers = data.setdefault("mcpServers", {})
    old = servers.get("ai_workforce")
    servers["ai_workforce"] = {"command": server_python(), "args": [str(Path(__file__).resolve())]}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  Claude Desktop config updated: {target}")
    if isinstance(old, dict) and old.get("env"):
        print("  (the old keys that were inside that config entry were removed; keys now live in aiw_config.json)")
    return True


def _ask_key(label, current):
    hint = "Enter = keep the saved one" if current else "Enter = skip"
    v = _getpass(f"  {label} ({hint}; typing is hidden, just paste and press Enter): ").strip().strip('"').strip("'")
    return v or current


def _setup_wizard():
    print("=" * 64)
    print(" Unstoppable AI Workforce v%s - setup" % __version__)
    print("=" * 64)
    print(" Please CLOSE Claude Desktop completely before continuing")
    print(" (otherwise it can overwrite the settings I am about to write).")
    print(" Your keys are stored only in: %s" % _config_path())
    print()
    cur = _read_json(_config_path(), {}) if _config_path().exists() else {}
    upd = {}
    print("1) Gemini keys (https://aistudio.google.com/apikey). A second key from another Google account doubles your free limits.")
    upd["gemini_key_a"] = _ask_key("Gemini key A", cur.get("gemini_key_a", ""))
    upd["gemini_key_b"] = _ask_key("Gemini key B", cur.get("gemini_key_b", ""))
    print("\n2) Creative worker: Groq (https://console.groq.com, key starts gsk_) or xAI Grok (https://console.x.ai, key starts xai-).")
    k = _ask_key("Groq / xAI key", cur.get("xai_api_key", ""))
    upd["xai_api_key"] = k
    if k and k != cur.get("xai_api_key"):
        if k.startswith("gsk_"):
            upd["xai_base_url"] = "https://api.groq.com/openai/v1"
            upd["grok_model"] = GROQ_DEFAULT[0]
            print("  Detected Groq key -> using Groq with model %s." % GROQ_DEFAULT[0])
        elif k.startswith("xai-"):
            upd["xai_base_url"] = "https://api.x.ai/v1"
            upd["grok_model"] = GROK_DEFAULT[0]
            print("  Detected xAI key -> using xAI Grok.")
    print("\n3) Project folders the workers may read and edit.")
    print("   Paste a folder that CONTAINS your projects (e.g. C:\\Users\\you\\Projects) or one project folder.")
    print("   Press Enter on an empty line when done. New projects are created in: %s" % workspace_dir())
    while True:
        p = _input("  Folder path (Enter = done): ").strip().strip('"').strip("'")
        if not p:
            break
        try:
            pp = Path(p).expanduser()
            if pp.is_dir() and is_too_broad(pp):
                if _input("  That folder is very broad (your whole user folder or a drive). Type YES to allow it anyway: ").strip() != "YES":
                    continue
            print("  Added: %s" % add_root(p))
        except ProjectError as e:
            print("  %s" % e)
    ar = _input("\n4) Allow the workers to RUN your tests/build commands (run_check)? They execute code on this PC. [y/N]: ").strip().lower()
    upd["allow_run"] = ar in ("y", "yes")
    save_config(upd)
    print("\n  Saved settings.")
    ic = _input("\n5) Add the server to Claude Desktop automatically? [Y/n]: ").strip().lower()
    if ic in ("", "y", "yes"):
        install_claude_config()
    else:
        print("  Skipped. Add manually (see README.md).")
    t = _input("\n6) Test the connections now? [Y/n]: ").strip().lower()
    if t in ("", "y", "yes"):
        _self_check()
    print("\nAll done. Start Claude Desktop again. In Settings > Developer you should see ai_workforce 'running'.")
    return 0


def _self_check():
    print("Unstoppable AI Workforce v%s - connectivity check" % __version__)
    print(f"  Gemini keys: {len(gemini_keys())}   Grok/Groq keys: {len(grok_keys())}   ({xai_base_url()})")
    print(f"  Config file: {_config_path()}  (exists: {_config_path().exists()})")
    print(f"  Workspace: {workspace_dir()}")
    print(f"  Allowed folders: " + "; ".join(str(r) for r in allowed_roots()))
    ok_all = True
    for worker in ("fast", "deep", "creative"):
        t0 = time.time()
        out = run_worker(worker, "Reply with the single word OK.")
        ok = isinstance(out, Result)
        ok_all &= ok
        print(f"\n[{worker}] {'PASS' if ok else 'FAIL'} in {time.time() - t0:.1f}s" + (f" via {out.handled_by}" if ok else ""))
        if not ok:
            print("  " + str(out).strip().replace("\n", "\n  ")[:700])
    if _GEM_BACKEND:
        print("\n  Gemini backend used per key: " + ", ".join(sorted(set(_GEM_BACKEND.values()))))
    if ok_all:
        print("\n[project engine] creating a test project, editing it, undoing it ...")
        try:
            proj = Project(workspace_dir() / "_aiw_check")
            (workspace_dir() / "_aiw_check").mkdir(parents=True, exist_ok=True)
            proj.refresh()
            ACTIVE["project"] = proj
            ok1, t1, c1 = _work_core(proj, "Create hello.py with a function add(a, b) returning a + b.", ["hello.py"], None, "fast", False, True)
            print("  create:", t1.splitlines()[0])
            ok2, t2, c2 = _work_core(proj, "Add a function sub(a, b) returning a - b to hello.py.", ["hello.py"], None, "fast", False, True)
            print("  edit:  ", t2.splitlines()[0])
            if ok2:
                _c, restored, conf = undo_change(proj, c2)
                print("  undo:   restored", restored, "conflicts", conf)
        except Exception as e:
            print("  project engine check failed:", e)
    print("\n" + status())
    return 0 if ok_all else 1


def _list_models():
    """Prints the model names each of your keys can use (to fix retired/renamed model names)."""
    print("Models your keys can use (copy names into aiw_config.json: flash_models, pro_models, grok_model)")
    seen = set()
    for label, key in gemini_keys():
        if key in seen:
            continue
        seen.add(key)
        try:
            from google import genai
            client = genai.Client(api_key=key)
            names = sorted(m.name.replace("models/", "") for m in client.models.list()
                           if "generateContent" in (getattr(m, "supported_actions", None) or ["generateContent"]))
            names = [n for n in names if n.startswith("gemini")]
            print(f"\nGemini ({label}):\n  " + "\n  ".join(names))
        except Exception as e:
            print(f"\nGemini ({label}): could not list models: {str(e)[:150]}")
    for label, key in grok_keys():
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key, base_url=xai_base_url(), timeout=30, max_retries=0)
            names = sorted(m.id for m in client.models.list().data)
            print(f"\n{xai_base_url()} ({label}):\n  " + "\n  ".join(names))
        except Exception as e:
            print(f"\n{xai_base_url()}: could not list models: {str(e)[:150]}")
    return 0


def _cli(argv):
    if "--list-models" in argv:
        return _list_models()
    if "--setup" in argv:
        return _setup_wizard()
    if "--check" in argv:
        return _self_check()
    if "--install-config" in argv:
        return 0 if install_claude_config() else 1
    if "--list-projects" in argv:
        print(list_projects_text())
        return 0
    if "--add-project" in argv:
        i = argv.index("--add-project")
        path = argv[i + 1] if len(argv) > i + 1 and not argv[i + 1].startswith("--") else _input("Paste the project folder path (or drag the folder here): ")
        try:
            p = Path(path.strip().strip('"').strip("'")).expanduser()
            if p.is_dir() and is_too_broad(p) and "--yes" not in argv:
                print("Refused: that is a drive root or your whole user folder. Pick a narrower folder (or add --yes).")
                return 1
            print("Added: %s" % add_root(path))
            print("No restart needed. In Claude: open_project(project='%s')" % p.name)
            return 0
        except ProjectError as e:
            print("Error: %s" % e)
            return 1
    if "--remove-project" in argv:
        i = argv.index("--remove-project")
        if len(argv) <= i + 1:
            print("Usage: --remove-project <path>")
            return 1
        print("Removed: %s" % remove_root(argv[i + 1]))
        return 0
    if "--set-check" in argv:
        i = argv.index("--set-check")
        if len(argv) < i + 4:
            print('Usage: --set-check <project folder name> <check name> "<command>"')
            return 1
        checks = dict(_cfg_data().get("checks") or {})
        checks.setdefault(argv[i + 1], {})[argv[i + 2]] = argv[i + 3]
        save_config({"checks": checks})
        print("Saved check '%s' for %s. (Also set allow_run via --setup.)" % (argv[i + 2], argv[i + 1]))
        return 0
    return None


if __name__ == "__main__":
    rc = _cli(sys.argv[1:])
    if rc is not None:
        sys.exit(rc)
    workspace_dir().mkdir(parents=True, exist_ok=True)
    data_dir().mkdir(parents=True, exist_ok=True)
    log(f"Starting Unstoppable AI Workforce v{__version__}... workspace: {workspace_dir()}")
    mcp.run()
