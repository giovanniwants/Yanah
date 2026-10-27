"""
Tool calling for mlx_server.py: Qwen3-Coder's XML tool-call format <-> OpenAI's.

Qwen3-Coder(-Next) writes a tool call as text:

    <tool_call>
    <function=write>
    <parameter=filePath>
    /tmp/app.py
    </parameter>
    <parameter=content>
    print("hi")
    </parameter>
    </function>
    </tool_call>

OpenAI-compatible clients (OpenCode, Aider, ...) need it as structured
`tool_calls` with JSON `arguments`. This module:

  * ToolCallStream: fed the model's text as it streams. Plain text passes
    straight through; from the first "<tool_call>" on, text is held and, at
    the end, parsed into OpenAI tool calls. Only used when the request
    carries `tools`, so ordinary chats (Yanah) are untouched.
  * parse_tool_calls: the parser (same rules as vLLM/SGLang's qwen3_coder
    parser: one leading and one trailing newline of a value are dropped;
    values are converted to the type the tool's JSON schema declares).
  * normalize_messages: the other direction. Clients send earlier tool calls
    back with `arguments` as a JSON string; Qwen's chat template iterates
    them as a mapping, so they are decoded to dicts first.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Dict, List, Optional, Tuple

START = "<tool_call>"
FUNC = "<function="        # if a tokenizer hides <tool_call> as a special token, calls start here
_BLOCK = re.compile(r"(?:<tool_call>\s*)?<function=([^>\n]+)>(.*?)(?:</function>\s*(?:</tool_call>|$)|$)", re.S)


def _first_marker(text: str) -> int:
    hits = [i for i in (text.find(START), text.find(FUNC)) if i >= 0]
    return min(hits) if hits else -1
_PARAM = re.compile(r"<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.S)


def _schemas(tools: Optional[list]) -> Dict[str, dict]:
    """{tool name: {param name: json-schema}} from an OpenAI `tools` list."""
    out = {}
    for t in tools or []:
        fn = (t or {}).get("function") or {}
        if fn.get("name"):
            out[fn["name"]] = ((fn.get("parameters") or {}).get("properties") or {})
    return out


def _convert(value: str, schema: Optional[dict]):
    """Text from the model -> the JSON type the tool expects."""
    types = (schema or {}).get("type")
    types = types if isinstance(types, list) else [types]
    if value.strip().lower() == "null" and "null" in types:
        return None
    for t in types:
        try:
            if t in ("integer", "int"):
                return int(value.strip())
            if t == "number":
                f = float(value.strip())
                return int(f) if f.is_integer() and "." not in value else f
            if t == "boolean":
                v = value.strip().lower()
                if v in ("true", "false"):
                    return v == "true"
            if t in ("object", "array"):
                return json.loads(value)
        except (ValueError, TypeError):
            continue
    if not schema or types == [None]:
        # unknown parameter: keep obvious JSON values (numbers, objects), else the text
        s = value.strip()
        if s[:1] in "[{" or s in ("true", "false", "null") or re.fullmatch(r"-?\d+(\.\d+)?", s or "x"):
            try:
                return json.loads(s)
            except ValueError:
                pass
    return value


def _strip_one_newline(v: str) -> str:
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


def parse_tool_calls(text: str, tools: Optional[list] = None) -> Tuple[str, List[dict]]:
    """-> (content before the first tool call, [OpenAI tool_call dicts]).
    If nothing parses, returns (text, []) so nothing is ever lost."""
    i = _first_marker(text)
    if i < 0:
        return text, []
    content, rest = text[:i], text[i:]
    schemas = _schemas(tools)
    calls = []
    for m in _BLOCK.finditer(rest):
        name = m.group(1).strip()
        args = {}
        for p in _PARAM.finditer(m.group(2)):
            pname = p.group(1).strip()
            args[pname] = _convert(_strip_one_newline(p.group(2)), schemas.get(name, {}).get(pname))
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    if not calls:
        return text, []
    return content.rstrip("\n") if content.strip() else "", calls


class ToolCallStream:
    """Streams plain text, holds everything from "<tool_call>" on."""

    def __init__(self, tools: Optional[list]):
        self.tools = tools
        self.buf = ""          # text not yet emitted
        self.holding = False   # True once "<tool_call>" has appeared

    def push(self, text: str) -> str:
        if self.holding:
            self.buf += text
            return ""
        self.buf += text
        i = _first_marker(self.buf)
        if i >= 0:
            out, self.buf, self.holding = self.buf[:i], self.buf[i:], True
            return out
        hold = 0                                  # a partial "<tool_c" / "<func" at the end might become one
        for marker in (START, FUNC):
            for n in range(min(len(marker) - 1, len(self.buf)), 0, -1):
                if self.buf.endswith(marker[:n]):
                    hold = max(hold, n)
                    break
        out = self.buf[:len(self.buf) - hold]
        self.buf = self.buf[len(out):]
        return out

    def finish(self) -> Tuple[str, List[dict]]:
        """-> (remaining text to show, tool calls)."""
        rest, self.buf = self.buf, ""
        if not self.holding:
            return rest, []
        content, calls = parse_tool_calls(rest, self.tools)
        return (content if calls else rest), calls


def normalize_messages(messages: List[dict]) -> List[dict]:
    """Earlier assistant tool calls: arguments JSON string -> dict (Qwen's template needs a mapping)."""
    out = []
    for m in messages:
        tcs = m.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            fixed = []
            for tc in tcs:
                tc = dict(tc or {})
                fn = dict(tc.get("function") or {})
                a = fn.get("arguments")
                if isinstance(a, str):
                    try:
                        a = json.loads(a) if a.strip() else {}
                    except ValueError:
                        a = {"input": a}
                fn["arguments"] = a if isinstance(a, dict) else {"input": a}
                tc["function"] = fn
                fixed.append(tc)
            m = dict(m, tool_calls=fixed)
        out.append(m)
    return out
