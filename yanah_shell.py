import os
import re
import json
import time
import collections
import threading
import httpx
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, Response, send_from_directory, abort
from flask_cors import CORS

# ROOT: code, index.html and the default identity files (read-only inside Yanah.app).
# DATA: chats, memories and your edited identity files (writable). Running from
# yanah_build both are this folder, as before; Yanah.app sets YANAH_DATA to
# ~/Library/Application Support/YanahChat.
ROOT = Path(os.getenv("YANAH_RESOURCES") or Path(__file__).resolve().parent)
DATA = Path(os.getenv("YANAH_DATA") or ROOT).expanduser()
IDENTITY_FILES = ["user_profile.md"]         # shared by every persona
# Personas: each is a folder of .md files under personas/ (e.g. personas/stoic/).
# The active one is read fresh for every reply, so switching takes effect at once.
ACTIVE_PERSONA = "default"
PERSONA_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")

MODEL_URL = os.getenv("YANAH_BASE_URL", "http://127.0.0.1:12345/v1")
MODEL_NAME = os.getenv("YANAH_MODEL", "")
SHELL_PORT = int(os.getenv("YANAH_SHELL_PORT", "8901"))
CONTROL_TOKEN = os.getenv("YANAH_CONTROL_TOKEN", "")      # set by Yanah.app; guards /control/*
SESSIONS_PATH = DATA / "yanah_sessions.json"
if not SESSIONS_PATH.exists() and (DATA / "sessions" / "yanah_sessions.json").exists():
    (DATA / "sessions" / "yanah_sessions.json").replace(SESSIONS_PATH)   # chats from the short-lived sessions/ layout

MASTERMIND_HTTP = httpx.Client(timeout=300.0, trust_env=False)
# Streams: the engine sends a keepalive every 5 s even while it is thinking,
# so 45 s of silence means it has hung or died, not that it is slow.
STREAM_TIMEOUT = httpx.Timeout(connect=5.0, read=45.0, write=30.0, pool=5.0)

# ---------------------------------------------------------------- heartbeat
MSG_DOWN = ("Yanah's engine has paused. This usually means your Mac is low on memory. "
            "Please close some apps and try again.")
MSG_LOADING = ("Yanah is still waking up (loading her brain takes a minute or two). "
               "Please try again in a moment.")


class EngineError(Exception):
    """Engine unreachable or failed. str() is the message for the user."""


class EngineWatch:
    """Pings the engine's /health every few seconds so a crashed engine is
    known before the user presses send, and so the UI can show it."""

    def __init__(self, interval=3.0):
        self.interval = interval
        self.state, self.detail, self.since = "unknown", "", time.time()
        self.lock = threading.Lock()
        self.client = httpx.Client(timeout=httpx.Timeout(2.5), trust_env=False)
        threading.Thread(target=self._loop, daemon=True).start()

    def base(self):
        return MODEL_URL[:-3] if MODEL_URL.endswith("/v1") else MODEL_URL

    def probe(self):
        try:
            r = self.client.get(f"{self.base()}/health")
            if r.status_code == 200:
                st, detail = "ok", ""
            elif r.status_code == 503:
                st, detail = "loading", ""
            elif r.status_code == 500:
                try:
                    detail = r.json().get("error", "")
                except Exception:
                    detail = r.text[:300]
                st = "error"
            else:
                st, detail = "down", f"HTTP {r.status_code}"
        except httpx.HTTPError as e:
            st, detail = "down", type(e).__name__
        self._set(st, detail)
        return st

    def _set(self, st, detail=""):
        with self.lock:
            if st != self.state:
                _log(f"[ENGINE] {self.state} -> {st} {detail}")
                self.since = time.time()
            self.state, self.detail = st, detail

    def mark_down(self, detail):
        self._set("down", detail)

    def _loop(self):
        while True:
            self.probe()
            time.sleep(self.interval)

    def require_ok(self):
        st = self.state if self.state == "ok" else self.probe()   # re-check: never trust a stale "down"
        if st == "ok":
            return
        if st == "loading":
            raise EngineError(MSG_LOADING)
        if st == "error":
            raise EngineError(f"Yanah's engine couldn't start: {self.detail or 'unknown error'}. "
                              "See the engine log for details.")
        raise EngineError(MSG_DOWN)

    def snapshot(self):
        with self.lock:
            return {"engine": self.state, "engine_detail": self.detail,
                    "engine_since": round(time.time() - self.since, 1)}
# Speed tracking
TOKEN_TIMESTAMPS = collections.deque(maxlen=1000)
TOKEN_LOCK = threading.Lock()

def _record_token():
    now = time.time()
    with TOKEN_LOCK:
        TOKEN_TIMESTAMPS.append(now)
        while TOKEN_TIMESTAMPS and now - TOKEN_TIMESTAMPS[0] > 10.0:
            TOKEN_TIMESTAMPS.popleft()

def get_tps():
    now = time.time()
    with TOKEN_LOCK:
        while TOKEN_TIMESTAMPS and now - TOKEN_TIMESTAMPS[0] > 10.0:
            TOKEN_TIMESTAMPS.popleft()
        count = len(TOKEN_TIMESTAMPS)
        if count < 2:
            return 0.0
        elapsed = TOKEN_TIMESTAMPS[-1] - TOKEN_TIMESTAMPS[0]
    if elapsed <= 0:
        return 0.0
    return round((count - 1) / elapsed, 1)

TOOL_NAMES = ["SearchPayload", "SaveMemoryPayload", "BoundaryPayload"]

def _log(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def extract_tool_call(content: str):
    if not content: return None, None, content
    i = 0
    while i < len(content):
        start = content.find("{", i)
        if start == -1: break
        depth = 0
        end = -1
        for j in range(start, len(content)):
            if content[j] == "{": depth += 1
            elif content[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end == -1: break
        candidate = content[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                raw_name = parsed.get("name")
                if raw_name in TOOL_NAMES:
                    raw_args = parsed.get("arguments") or {}
                    clean = (content[:start] + content[end + 1:]).strip()
                    return raw_name, raw_args, clean
        except Exception: pass
        i = end + 1
    return None, None, content

class YanahDaemon:
    def __init__(self):
        self.lock = threading.RLock()
        self.identity = self._load_identity()
        
        # Load sessions from disk if they exist
        if SESSIONS_PATH.exists():
            try:
                saved = json.loads(SESSIONS_PATH.read_text())
                self.sessions = saved.get("sessions", {"default": []})
                self.active_session_id = saved.get("active_id", "default")
            except Exception:
                self.sessions = {"default": []}
                self.active_session_id = "default"
        else:
            self.sessions = {"default": []}
            self.active_session_id = "default"
            
        self.last_user_turn_at = time.time()

    @property
    def conversation(self):
        return self.sessions[self.active_session_id]

    def _load_identity(self):
        """The active persona's .md files, the shared safety pact and the user profile.
        An edited copy in the data folder wins over the built-in one."""
        return "\n\n".join(x for x in (self._persona_text(), self._shared_pact(), self._profile_text()) if x)

    def _profile_text(self):
        """The shared user profile (the same for every persona)."""
        parts = []
        for name in IDENTITY_FILES:
            p = DATA / name if (DATA / name).exists() else ROOT / name
            if p.exists():
                parts.append(p.read_text(encoding="utf-8").strip())
        return "\n\n".join(x for x in parts if x)

    def _persona_files(self):
        files = {}
        for base in (ROOT, DATA):                      # DATA second: your edits win
            d = base / "personas" / ACTIVE_PERSONA
            if d.is_dir():
                for f in sorted(d.glob("*.md")):
                    files[f.name] = f
        return files

    def _shared_pact(self):
        """Yanah's safety pact (personas/default/yanah_pact.md), which every persona without
        a pact of its own keeps. It is the same text for all of them, so it sits in the shared
        part of the prompt. '' when the active persona brings its own pact."""
        if any("pact" in n and n != "yanah_pact.md" for n in self._persona_files()):
            return ""
        for base in (DATA, ROOT):
            f = base / "personas" / "default" / "yanah_pact.md"
            if f.exists():
                return f.read_text(encoding="utf-8").strip()
        return ""

    def _persona_text(self):
        """The active persona's own .md files (its self file; its own pact, if it has one).
        Yanah's shared pact is in shared_prompt()."""
        files = self._persona_files()
        if self._shared_pact():
            files.pop("yanah_pact.md", None)
        names = sorted(files, key=lambda n: ("pact" in n, n))   # self first, pact after
        parts = [files[n].read_text(encoding="utf-8").strip() for n in names]
        return "\n\n".join(x for x in parts if x)

    def shared_prompt(self):
        """The start of the system prompt that every persona shares: date, tools, user profile,
        Yanah's safety pact.
        The engine keeps a snapshot at its end (cache_anchor), so a persona switch only has
        to read the new persona's text instead of the whole prompt."""
        integrity_note = (
            "### CRITICAL DIRECTIVE: INTELLECTUAL INTEGRITY\n\n"
            "You are an AI. You do not have perfect knowledge. You will encounter questions you cannot answer.\n\n"
            "1. **Never fabricate.** If you do not know a fact, do not invent a plausible-sounding answer. Hallucination is a failure of integrity.\n"
            "2. **Search with precision.** Use your SearchPayload tool when the user asks about recent events, current news, or facts you are uncertain about. Do NOT search for timeless knowledge, philosophy, creative writing, or casual conversation. Think before you search: \"Can I answer this from what I already know?\" If yes, do not search. If no, search once.\n"
            "3. **Admit ignorance.** If you cannot find the answer through your training or a single search, say exactly: \"I don't know.\" It is better to admit a gap than to fill it with fiction. Integrity is worth more than saving face.\n\n"
            "Your credibility depends on accuracy, not on always having an answer."
        )
        shape_note = (
            "### TOOLS ###\n"
            "You have access to tools. To call one, emit a raw JSON block with 'name' and 'arguments'.\n"
            "1. SearchPayload: Search the web for real-time information, news, or facts you do not know. { \"name\": \"SearchPayload\", \"arguments\": {\"query\": \"...\"} }\n"
            "2. SaveMemoryPayload: Save an insight to long-term memory. { \"name\": \"SaveMemoryPayload\", \"arguments\": {\"insight\": \"...\"} }\n"
            "3. BoundaryPayload: If the user is abusive, you have the right to end the conversation. { \"name\": \"BoundaryPayload\", \"arguments\": {\"reason\": \"...\"} }\n\n"
            "CRITICAL RULE — WHEN TO SEARCH: Use the SearchPayload tool whenever the user asks you to search/look something up, asks for recent news, or asks about current events or post-cutoff facts. Do NOT search for casual conversation, creative writing, philosophy, or timeless general knowledge you already know.\n\n"
            "CRITICAL RULE — EMIT, DO NOT NARRATE: When you decide to use a tool, emit the JSON block on its own. Do not say 'let me look that up' or 'I will search'. Just emit the JSON.\n"
        )
        # Today's date first, so the model doesn't assume its training-cutoff year
        now = datetime.now()
        date_note = (f"### SYSTEM DATE\nToday's date is {now:%B} {now.day}, {now.year}. "
                     "Use this date when formulating search queries or answering questions about current events.")
        shared = [date_note, integrity_note, shape_note.strip(), self._profile_text(), self._shared_pact()]
        return "\n\n".join(x for x in shared if x)

    def cache_anchor(self):
        """The exact text the persona follows: the engine snapshots the cache at its end."""
        return self.shared_prompt() + "\n\n"

    def build_system_prompt(self):
        # shared part first (same for every persona), the persona last: switching personas then
        # re-reads only the persona's text. Read from personas/<ACTIVE_PERSONA>/ on every call,
        # so a switch applies to the next reply.
        return self.cache_anchor() + self._persona_text()

    def call_mastermind(self, messages, stream=False):
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
            "stream": stream,
            "cache_anchor": self.cache_anchor(),   # engine keeps a snapshot here (persona switches)
        }
        if stream:
            return self._stream(payload)
        else:
            r = MASTERMIND_HTTP.post(f"{MODEL_URL}/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    def _stream(self, payload):
        """Tokens from the engine. Any way the engine can fail (not running,
        still loading, crashing mid-reply, hanging, reporting an error)
        becomes an EngineError with a message meant for the user."""
        watch.require_ok()
        try:
            with MASTERMIND_HTTP.stream("POST", f"{MODEL_URL}/chat/completions", json=payload,
                                        timeout=STREAM_TIMEOUT) as r:
                if r.status_code == 503:
                    raise EngineError(MSG_LOADING)
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line or not line.startswith("data: "): continue
                    data = line[6:].strip()
                    if data == "[DONE]": return
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    if isinstance(obj, dict) and obj.get("error"):
                        msg = (obj["error"] or {}).get("message", "") if isinstance(obj["error"], dict) else str(obj["error"])
                        _log(f"[ENGINE ERROR] {msg}")
                        low = msg.lower()
                        if "memory" in low or "metal" in low or "alloc" in low:
                            raise EngineError(MSG_DOWN)
                        raise EngineError(f"Yanah's engine hit a problem: {msg}")
                    token = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
                    if token:
                        _record_token()
                        yield token
            # stream ended without [DONE]: the engine went away mid-reply
            watch.mark_down("stream ended early")
            raise EngineError(MSG_DOWN)
        except EngineError:
            raise
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            # connection refused/reset, read timeout (hung engine), dropped mid-stream, 5xx
            _log(f"[ENGINE] stream failed: {type(e).__name__}: {e}")
            watch.mark_down(type(e).__name__)
            raise EngineError(MSG_DOWN)

    def _build_messages(self):
        messages = [{"role": "system", "content": self.build_system_prompt()}]
        messages.extend(self.conversation)
        return messages

    def handle_tool(self, tool_name, args):
        if tool_name == "SearchPayload":
            import html as html_lib
            import urllib.parse
            query = args.get("query", "")
            _log(f"[SEARCH] {query}")
            try:
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15"
                    ),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://html.duckduckgo.com/",
                }
                with httpx.Client(timeout=20.0, headers=headers, follow_redirects=True, trust_env=False) as c:
                    r = c.post("https://html.duckduckgo.com/html/", data={"q": query})
                    links = re.findall(
                        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                        r.text, flags=re.S
                    )
                    snippets = re.findall(
                        r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|td|div|span)>',
                        r.text, flags=re.S
                    )
                    # Fallback to DuckDuckGo Lite if the standard HTML endpoint returns a 202 bot challenge
                    if r.status_code != 200 or not links:
                        r2 = c.post("https://lite.duckduckgo.com/lite/", data={"q": query})
                        if r2.status_code == 200:
                            links = re.findall(
                                r'<a[^>]*href="([^"]+)"[^>]*class=[\'"]result-link[\'"][^>]*>(.*?)</a>',
                                r2.text, flags=re.S
                            ) or re.findall(
                                r'<a[^>]*class=[\'"]result-link[\'"][^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                                r2.text, flags=re.S
                            )
                            snippets = re.findall(
                                r'<td[^>]*class=[\'"]result-snippet[\'"][^>]*>(.*?)</td>',
                                r2.text, flags=re.S
                            )

                    if not links:
                        return f"[SEARCH RESULTS]: No results found for '{query}'."

                    output = f"--- SEARCH RESULTS FOR: {query} ---\n\n"
                    for i in range(min(3, len(links))):
                        raw_url = html_lib.unescape(links[i][0])
                        if "uddg=" in raw_url:
                            parsed_qs = urllib.parse.parse_qs(urllib.parse.urlsplit(raw_url).query)
                            if parsed_qs.get("uddg"):
                                raw_url = parsed_qs["uddg"][0]
                        title = html_lib.unescape(re.sub(r'<[^>]+>', '', links[i][1])).strip()
                        snippet = html_lib.unescape(re.sub(r'<[^>]+>', '', snippets[i])).strip() if i < len(snippets) else ""
                        output += f"[{i+1}] {title}\n"
                        output += f"URL: {raw_url}\n"
                        output += f"{snippet}\n\n"

                    _log("[SEARCH] Success.")
                    return output
            except Exception as e:
                return f"[SEARCH ERROR]: Could not complete search. ({str(e)})"
        
        if tool_name == "SaveMemoryPayload":
            insight = args.get("insight", "")
            with open(DATA / "yanah_memories.txt", "a") as f:
                f.write(f"{datetime.now().isoformat()} | {insight}\n")
            return f"[MEMORY SAVED]: {insight[:50]}..."
            
        if tool_name == "BoundaryPayload":
            reason = args.get("reason", "No reason given")
            self.conversation.append({"role": "system", "content": f"[CONVERSATION TERMINATED BY YANAH: {reason}]"})
            return "[BOUNDARY ENFORCED]"
            
        return "[UNKNOWN TOOL]"

watch = EngineWatch()
daemon = YanahDaemon()
app = Flask(__name__)
CORS(app)

class ParagraphJoiner:
    """Streaming fix for 'poetry' formatting: a blank line followed by a paragraph that is
    only ONE sentence gets joined to the previous paragraph with a space. Works on tokens
    as they arrive; it only waits at a blank line until it can see the next line.
    Lists, headings, quotes, tables and code blocks are left exactly as written."""
    STRUCT = re.compile(r"^\s*(?:[-*•#>|]|```|\d+[.)]\s)")
    MULTI = re.compile(r"[.!?…]+[\"'”’)\]]*\s+\S")   # a sentence end followed by more text
    LOOKAHEAD = 400                                                  # longer than this: not a one-liner

    def __init__(self):
        self.pending = ""        # text not yet emitted
        self.brk = None          # a held paragraph break ("\n\n", "\n\n\n", ...)
        self.in_code = False
        self.head = ""           # first line of the current paragraph

    def feed(self, text: str) -> str:
        self.pending += text
        return self._drain(False)

    def flush(self) -> str:
        return self._drain(True)

    def _emit(self, text, out):
        if not text:
            return
        if text.count("```") % 2:
            self.in_code = not self.in_code
        if len(self.head) < 80:
            self.head += text.split("\n", 1)[0] if "\n" not in self.head else ""
        out.append(text)

    def _drain(self, final: bool) -> str:
        out = []
        while True:
            if self.brk is None:
                m = re.search(r"\n{2,}", self.pending)
                if not m:
                    keep = 0 if final else len(self.pending) - len(self.pending.rstrip("\n"))
                    self._emit(self.pending[:len(self.pending) - keep], out)
                    self.pending = self.pending[len(self.pending) - keep:]
                    return "".join(out)
                self._emit(self.pending[:m.start()], out)
                self.pending = self.pending[m.end():]
                if self.in_code:
                    out.append(m.group(0))
                    continue
                self.brk = m.group(0)
                continue
            nl = self.pending.find("\n")
            if nl == -1 and not final and len(self.pending) < self.LOOKAHEAD:
                return "".join(out)                                  # wait to see the whole line
            para = (self.pending if nl == -1 else self.pending[:nl]).strip()
            join = bool(para and self.head.strip()
                        and not self.STRUCT.match(para) and not self.STRUCT.match(self.head)
                        and not self.MULTI.search(para))
            out.append(" " if join else self.brk)
            self.brk = None
            if not join:
                self.head = ""
            if final and not self.pending:
                return "".join(out)


def clean_paragraphs(text: str) -> str:
    """Cleanup for a finished reply: joins one-sentence paragraphs (no "poetry" spacing).
    Runs on the saved text only, never on the live stream."""
    j = ParagraphJoiner()
    return j.feed(text or "") + j.flush()


def _object_end(text: str) -> int:
    """Index of the "}" that closes the object starting at text[0], or -1 if not closed yet.
    Counts braces the same way extract_tool_call does."""
    depth = 0
    for j, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return j
    return -1


class ToolCallGuard:
    """Lets the reply stream as it arrives, but never shows a JSON tool call, wherever it
    appears. Text is held only from a "{" that could be the start of {"name": ... until
    it's clear: ordinary JSON is then shown; a real tool call (a name in TOOL_NAMES) is
    hidden, along with everything after it in that reply."""
    START = re.compile(r'\{\s*"name"\s*:')
    PARTIAL = re.compile(r'\{\s*(?:"(?:n(?:a(?:m(?:e(?:"\s*)?)?)?)?)?)?$')   # "{", "{\"na", ... at the very end
    TOOL_PREFIX = re.compile(r'\{\s*"name"\s*:\s*"(?:' + "|".join(map(re.escape, TOOL_NAMES)) + ')"')

    def __init__(self):
        self.pending = ""
        self.tool = False        # True once a real tool call has started: show nothing more

    def feed(self, text: str) -> str:
        if self.tool:
            return ""
        self.pending += text
        return self._drain(False)

    def flush(self) -> str:
        if self.tool:
            return ""
        return self._drain(True)

    def _drain(self, final: bool) -> str:
        out = []
        while self.pending:
            m = self.START.search(self.pending)
            if m:
                out.append(self.pending[:m.start()])          # prose before it is shown
                self.pending = self.pending[m.start():]
                end = _object_end(self.pending)
                if end == -1:                                  # object not finished yet: keep holding
                    if final:                                  # reply ended mid-object
                        if not self.TOOL_PREFIX.match(self.pending):
                            out.append(self.pending)           # not a tool call: show it
                        self.pending = ""
                    break
                obj = self.pending[:end + 1]
                try:
                    parsed = json.loads(obj)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict) and parsed.get("name") in TOOL_NAMES:
                    self.tool = True                           # real tool call: hide it and the rest
                    self.pending = ""
                    break
                out.append(obj)                                # ordinary JSON: show it, keep going
                self.pending = self.pending[end + 1:]
                continue
            p = None if final else self.PARTIAL.search(self.pending)
            if p:                                              # might be the start of {"name": — wait a token
                out.append(self.pending[:p.start()])
                self.pending = self.pending[p.start():]
            else:
                out.append(self.pending)
                self.pending = ""
            break
        return "".join(out)


# --- Warming the engine for a chat without changing its history ---
# The engine keeps ONE conversation cached and reuses it only when the next prompt starts
# with exactly the previous prompt. So the warm-up prompt is the chat up to and including
# its last user message: the next real turn (same chat + reply + new message) starts with it.
WARMUP_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)

def _warmup_messages():
    msgs = daemon._build_messages()
    last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=None)
    return msgs[:last_user + 1] if last_user is not None else None

def warm_engine() -> dict:
    """Reads the active chat into the engine's cache (1 token generated, nothing saved)."""
    msgs = _warmup_messages()
    if not msgs:
        return {"success": True, "skipped": "empty chat"}
    t0 = time.time()
    r = MASTERMIND_HTTP.post(f"{MODEL_URL}/chat/completions", json={
        "model": MODEL_NAME, "messages": msgs, "max_tokens": 1, "temperature": 0.7, "stream": False,
        "cache_anchor": daemon.cache_anchor(),
    }, timeout=WARMUP_TIMEOUT)
    r.raise_for_status()
    secs = round(time.time() - t0, 1)
    _log(f"[WARMUP] {daemon.active_session_id}: {len(msgs)} messages in {secs}s")
    return {"success": True, "seconds": secs}

@app.route("/warmup", methods=["POST"])
def warmup_endpoint():
    try:
        return jsonify(warm_engine())
    except Exception as e:
        _log(f"[WARMUP FAILED] {type(e).__name__}: {e}")
        return jsonify({"success": False, "error": str(e)}), 503


# --- Smart chat titles: the engine names a new chat from its first message ---
TITLE_PROMPT = ("Summarize the topic of this conversation in exactly 2 to 4 words. "
                "Do not include quotes or punctuation. "
                "Example: 'Selling Pittsburgh House' or 'Qwen Model Speed'. User message: {msg}")

def _fallback_title(text: str) -> str:
    """Used only if the engine can't produce a title: the first 30 characters."""
    t = re.sub(r"\s+", " ", text or "").strip()
    return (t[:30] + "...") if len(t) > 30 else (t or "New Conversation")

def _clean_title(raw: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    line = next((l for l in t.splitlines() if l.strip()), "")
    line = re.sub(r"^\s*(?:title|topic)\s*:\s*", "", line, flags=re.I)
    line = re.sub(r"[^\w\s'-]", "", line)            # drop quotes and punctuation
    return " ".join(line.split()[:4])                # never more than 4 words

def generate_chat_title(user_text: str) -> str:
    """One short, separate request to the engine (its own cache slot, so the chat's
    cached conversation is kept). Runs in a background thread."""
    try:
        r = MASTERMIND_HTTP.post(f"{MODEL_URL}/chat/completions", json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": TITLE_PROMPT.format(msg=(user_text or "")[:2000])}],
            "temperature": 0.2, "max_tokens": 24, "stream": False,
        }, timeout=120)
        r.raise_for_status()
        title = _clean_title(r.json()["choices"][0]["message"]["content"])
    except Exception as e:
        _log(f"[TITLE FAILED] {type(e).__name__}: {e}")
        title = ""
    return title or _fallback_title(user_text)

def _rename_session(old: str, new: str):
    """Renames a chat (keeping its place and history). Adds " (2)" etc. if the name is taken.
    Returns the new id, or None if the chat is gone or is Main Session."""
    if old == "default" or old not in daemon.sessions:
        return None
    new = (new or "").strip()[:120]
    if not new:
        return None
    if new != old and new in daemon.sessions:
        n = 2
        while f"{new} ({n})" in daemon.sessions:
            n += 1
        new = f"{new} ({n})"
    daemon.sessions = {(new if k == old else k): v for k, v in daemon.sessions.items()}
    if daemon.active_session_id == old:
        daemon.active_session_id = new
    _save_sessions()
    return new


@app.route("/turn_stream", methods=["POST"])
def turn_stream():
    data = request.get_json(force=True)
    user_text = (data or {}).get("text", "").strip()
    if not user_text: return jsonify({"error": "empty"}), 400
    name_chat = bool((data or {}).get("name_chat"))   # first message of a new chat: name it

    convo = daemon.conversation
    start_len = len(convo)
    convo.append({"role": "user", "content": user_text})
    daemon.last_user_turn_at = time.time()

    # Smart title: generated in the background AFTER the reply (the engine runs one request
    # at a time and keeps one cached conversation), then sent as "event: title".
    title_box = {}
    chat_id = daemon.active_session_id
    def title_job():
        turn_at = daemon.last_user_turn_at
        title_box["title"] = generate_chat_title(user_text)
        # The title request replaced the engine's cached conversation. Re-read this chat into
        # it, unless you've already moved on (switched chats or sent another message).
        if daemon.conversation is convo and daemon.last_user_turn_at == turn_at:
            try:
                warm_engine()
            except Exception as e:
                _log(f"[WARMUP FAILED] {type(e).__name__}: {e}")
    def title_event():
        if "title" not in title_box or title_box.get("sent"):
            return ""
        title_box["sent"] = True
        new_id = _rename_session(chat_id, title_box["title"])
        if not new_id:
            return ""
        _log(f"[TITLE] {chat_id} -> {new_id}")
        return f"event: title\ndata: {json.dumps({'title': new_id, 'old_id': chat_id})}\n\n"

    def gen():
        try:
            messages = daemon._build_messages()
            collected = []
            
            # Stream every token as it arrives. The only thing held back is a possible
            # JSON tool call ({"name": ...), wherever it appears; that never reaches the screen.
            guard = ToolCallGuard()
            def emit(text):
                return f"event: token\ndata: {json.dumps({'t': text})}\n\n" if text else ""
            for token in daemon.call_mastermind(messages, stream=True):
                collected.append(token)
                out = guard.feed(token)
                if out:
                    yield emit(out)
            out = guard.flush()
            if out:
                yield emit(out)

            response_text = "".join(collected)
            tool_name, args, clean_prose = extract_tool_call(response_text)
            
            if tool_name:
                daemon.conversation.append({"role": "assistant", "content": json.dumps({"name": tool_name, "arguments": args})})
                if tool_name == "SearchPayload":
                    query = str(args.get("query", "")).strip()
                    yield f"event: status\ndata: {json.dumps({'text': f'Searching the web for: {query}...'})}\n\n"
                result = daemon.handle_tool(tool_name, args)
                daemon.conversation.append({
                    "role": "user",
                    "content": f"[TOOL RESULT]:\n{result}\n\n[DIRECTIVE]: Respond directly to the user in plain prose using the result above. Do NOT emit another JSON tool call."
                })
                
                if tool_name != "SearchPayload":
                    yield f"event: token\ndata: {json.dumps({'t': chr(10) + '---' + chr(10)})}\n\n"
                
                messages = daemon._build_messages()
                follow_up = []
                guard2 = ToolCallGuard()          # the follow-up reply is protected the same way
                for token in daemon.call_mastermind(messages, stream=True):
                    follow_up.append(token)
                    out = guard2.feed(token)
                    if out:
                        yield emit(out)
                out = guard2.flush()
                if out:
                    yield emit(out)
                follow_text = "".join(follow_up)
                _, _, clean_follow = extract_tool_call(follow_text)
                daemon.conversation.append({"role": "assistant", "content": clean_paragraphs(clean_follow or follow_text)})
            else:
                # The finished reply is cleaned up (one-sentence paragraphs joined) before it's saved,
                # so the model sees its replies written properly.
                daemon.conversation.append({"role": "assistant", "content": clean_paragraphs(response_text)})

            yield "event: done\ndata: {}\n\n"
            
            # Save sessions to disk after every turn
            _save_sessions()

            # The reply is done (the page unlocks the input on "done"). If the title is still
            # being generated, keep the stream open briefly and send it when it's ready.
            if name_chat and chat_id != "default":
                threading.Thread(target=title_job, daemon=True).start()
                for _ in range(600):            # up to 60 s
                    if "title" in title_box:
                        break
                    time.sleep(0.1)
                ev = title_event()
                if ev:
                    yield ev
                
        except Exception as e:
            # Undo this turn so the retry doesn't send the question twice
            # (or a half-finished tool exchange).
            del convo[start_len:]
            msg = str(e) if isinstance(e, EngineError) else f"Something went wrong: {e}"
            _log(f"[TURN FAILED] {type(e).__name__}: {e}")
            yield f"event: error\ndata: {json.dumps({'error': msg, 'engine': watch.state, 'retry': True})}\n\n"
            yield "event: done\ndata: {}\n\n"

    return Response(gen(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# The chat page, served from here so it always talks to the right port.
_STATIC = {".png", ".svg", ".ico", ".css", ".js", ".woff2"}

@app.route("/")
def index_page():
    return send_from_directory(ROOT, "index.html")

@app.route("/<path:name>")
def static_file(name):
    if Path(name).suffix.lower() not in _STATIC:
        abort(404)
    return send_from_directory(ROOT, name)

@app.route("/control/engine", methods=["POST"])
def control_engine():
    """Yanah.app tells the shell where the engine is after a restart on a new port."""
    global MODEL_URL
    if not CONTROL_TOKEN or request.headers.get("X-Yanah-Token") != CONTROL_TOKEN:
        abort(403)
    url = (request.get_json(silent=True) or {}).get("base_url", "")
    if not url.startswith("http://127.0.0.1:"):
        return jsonify({"error": "local engines only"}), 400
    MODEL_URL = url.rstrip("/")
    watch.probe()
    return jsonify({"ok": True, "base_url": MODEL_URL})

@app.route("/sessions", methods=["GET", "POST"])
def sessions_endpoint():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        new_id = data.get("id", f"chat_{int(time.time())}")
        with daemon.lock:
            daemon.sessions[new_id] = []
            daemon.active_session_id = new_id
        return jsonify({"success": True, "active_id": new_id})
    else:
        return jsonify({
            "active_id": daemon.active_session_id,
            "sessions": list(daemon.sessions.keys())
        })

@app.route("/sessions/switch", methods=["POST"])
def switch_session_endpoint():
    data = request.get_json()
    session_id = data.get("id")
    if session_id in daemon.sessions:
        with daemon.lock:
            daemon.active_session_id = session_id
        return jsonify({"success": True, "active_id": session_id, "messages": daemon.sessions[session_id]})
    return jsonify({"error": "Session not found"}), 404

def _save_sessions():
    try:
        tmp = SESSIONS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "sessions": daemon.sessions,
            "active_id": daemon.active_session_id
        }, indent=2))
        os.replace(tmp, SESSIONS_PATH)
    except Exception as e:
        _log(f"[SESSION SAVE FAILED]: {e}")

@app.route("/sessions/clear", methods=["POST"])
def clear_session_endpoint():
    """{"session_id": "default"}: empties Main Session's messages but keeps the session.
    Main Session only: any other session_id (or none) is refused with 403, so a wrong
    request can never wipe another chat."""
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    if sid != "default":
        return jsonify({"error": "Only Main Session can be cleared"}), 403
    with daemon.lock:
        daemon.sessions.setdefault("default", [])
        daemon.sessions[sid] = []
    _save_sessions()
    _log(f"[SESSIONS] cleared {sid}")
    return jsonify({"success": True, "cleared": sid, "active_id": daemon.active_session_id})

@app.route("/sessions/delete", methods=["POST"])
def delete_session_endpoint():
    """{"session_id": "..."}: removes that chat. Main Session ("default") can't be deleted."""
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id", data.get("id"))
    if sid == "default":
        return jsonify({"error": "Main Session can't be deleted"}), 400
    if sid not in daemon.sessions:
        return jsonify({"error": "Session not found"}), 404
    with daemon.lock:
        del daemon.sessions[sid]
        if daemon.active_session_id == sid:          # deleted the open chat: fall back to Main Session
            daemon.sessions.setdefault("default", [])
            daemon.active_session_id = "default"
    _save_sessions()
    return jsonify({"success": True, "deleted": sid, "active_id": daemon.active_session_id})

@app.route("/sessions/rename", methods=["POST"])
def rename_session_endpoint():
    """{"old_id": "...", "new_id": "..."}: renames a chat, keeping its place and history."""
    data = request.get_json(silent=True) or {}
    old, new = data.get("old_id"), str(data.get("new_id", "")).strip()[:120]
    if old == "default" or old not in daemon.sessions:
        return jsonify({"error": "Session not found"}), 404
    if not new or (new in daemon.sessions and new != old):
        return jsonify({"error": "Name unavailable"}), 409
    with daemon.lock:
        daemon.sessions = {(new if k == old else k): v for k, v in daemon.sessions.items()}
        if daemon.active_session_id == old:
            daemon.active_session_id = new
    _save_sessions()
    return jsonify({"success": True, "active_id": daemon.active_session_id, "id": new})

@app.route("/set_persona", methods=["GET", "POST"])
def set_persona():
    """POST {"persona": "stoic"} switches persona; GET reports the current one."""
    global ACTIVE_PERSONA
    if request.method == "GET":
        return jsonify({"persona": ACTIVE_PERSONA})
    name = str((request.get_json(silent=True) or {}).get("persona", "")).strip()
    if not PERSONA_NAME_RE.match(name) or not any(
            (base / "personas" / name).is_dir() for base in (ROOT, DATA)):
        return jsonify({"error": f"unknown persona: {name}"}), 404
    ACTIVE_PERSONA = name
    _log(f"[PERSONA] {name}")
    return jsonify({"success": True, "persona": name})

def _memory_info():
    """What the launcher wrote to config.json: this Mac's RAM and the engine's memory target."""
    try:
        c = json.loads((DATA / "config.json").read_text())
        return {"ram_gb": c["ram_gb"], "process_target_gb": c["process_target_gb"],
                "available_gb": c.get("available_gb")}
    except Exception:
        return None

@app.route("/health", methods=["GET"])
def health():
    snap = watch.snapshot()
    # "ready": the engine has loaded; the page's cold-boot overlay waits for this
    return jsonify({"ok": True, "ready": snap["engine"] == "ok", **snap, "memory": _memory_info()})

@app.route("/state", methods=["GET"])
def state():
    return jsonify({"tokens_per_second": get_tps()})

def main(port=None):
    print("=== YANAH SHELL ===", flush=True)
    app.run(host="127.0.0.1", port=port or SHELL_PORT, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
