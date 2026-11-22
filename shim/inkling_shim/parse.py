"""Reading the model's output, token by token.

The assistant turn is a list of blocks (chat_template.jinja; llama.cpp common/parsers/inkling.cpp):
    <|message_model|><|content_thinking|> reasoning <|end_message|>
    <|message_model|><|content_text|> answer <|end_message|>
    <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>
    <|content_model_end_sampling|>
Every marker is a single special token, so the parser works on ids: a marker can never arrive half-written. Text is
decoded from the tokens' bytes with an incremental UTF-8 decoder, so a character split across tokens is held back
until it is whole. `StopMatcher` holds back answer text that could be the start of a client `stop` string.
"""
from __future__ import annotations

import codecs
import json
import re

from . import vocab as V

_ARGS_HEAD = re.compile(r'\s*\{\s*"name"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*"args"\s*:\s*')
_HEAD_GIVE_UP = 512        # chars of a tool body without the expected head: stop streaming it, send it whole at the end


class ToolCall:
    def __init__(self, index: int, name: str, kind: str):
        self.index, self.name, self.kind = index, name, kind
        self.body = ""             # the raw block text
        self.args_start = None     # where the "args" value starts inside body (JSON calls only)
        self.sent = 0              # chars of the args value already streamed
        self.arguments = None      # final arguments string (set when the block closes)
        self.args = None           # final arguments as an object (None: not valid JSON)

    def streamable(self) -> str:
        """Args text that is safe to stream now: everything but the last non-blank char (maybe the outer '}')."""
        if self.args_start is None:
            if self.kind == "json" and len(self.body) <= _HEAD_GIVE_UP + len(self.name):
                m = _ARGS_HEAD.match(self.body)
                if m:
                    self.args_start = m.end()
            if self.args_start is None:
                return ""
        rest = self.body[self.args_start:].rstrip()
        safe = max(0, len(rest) - 1)
        out = rest[self.sent:safe]
        self.sent = max(self.sent, safe)
        return out

    def close(self) -> str:
        """Finish the call; returns the args text not streamed yet."""
        name = self.name
        if self.args_start is not None:
            text = self.body[self.args_start:].rstrip()
            if text.endswith("}"):
                text = text[:-1]
            try:
                self.args = json.loads(text)
            except ValueError:
                self.args = None
            self.arguments = text
        else:
            obj = None
            try:
                obj = json.loads(self.body)
            except ValueError:
                pass
            if obj is None:                         # a damaged head ('{"name"""f","args":{...}}'): find the args
                m = re.search(r'"args"\s*:\s*', self.body)
                if m:
                    try:
                        args, _ = json.JSONDecoder().raw_decode(self.body, m.end())
                        obj = {"args": args}
                    except ValueError:
                        pass
            if isinstance(obj, dict):
                name = obj.get("name") if isinstance(obj.get("name"), str) and obj.get("name") else name
                args = obj.get("args", obj.get("arguments", {k: v for k, v in obj.items() if k != "name"}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {"raw": args}
                self.args = args if isinstance(args, dict) else {"value": args}
            else:
                self.args = {"input" if self.kind == "text" else "raw": self.body.strip()}
            self.arguments = json.dumps(self.args, ensure_ascii=False)
        if not self.name:
            self.name = name
        out = self.arguments[self.sent:]          # what was streamed is always a prefix of the final text
        self.sent = len(self.arguments)
        return out


class OutputParser:
    """feed(id) -> events:
        ("reasoning", text) | ("content", text)
        ("tool_begin", ToolCall) | ("tool_args", ToolCall, text) | ("tool_end", ToolCall)
    """

    BODY = {V.CONTENT_THINKING: "think", V.CONTENT_TEXT: "text", V.CONTENT_TOOL_ERROR: "text",
            V.CONTENT_TOOL_JSON: "tool", V.CONTENT_TOOL_TEXT: "tool", V.CONTENT_XML: "tool"}

    def __init__(self, vocab: V.Vocab):
        self.vocab = vocab
        self.state = None          # None (between blocks) | "header" | "think" | "text" | "tool" | "done"
        self.header = b""
        self.utf8 = codecs.getincrementaldecoder("utf-8")("replace")
        self.content = ""          # all answer text so far
        self.reasoning = ""
        self.calls: list[ToolCall] = []
        self.call: ToolCall | None = None

    @property
    def pending_bytes(self) -> bool:
        return bool(self.utf8.getstate()[0])

    def _text(self, raw: bytes, final=False) -> str:
        return self.utf8.decode(raw, final)

    def _emit_text(self, s: str, out: list) -> None:
        if not s:
            return
        if self.state == "think":
            self.reasoning += s
            out.append(("reasoning", s))
        elif self.state == "text":
            self.content += s
            out.append(("content", s))
        elif self.state == "tool":
            self.call.body += s
            chunk = self.call.streamable() if self.call.kind == "json" else ""
            if chunk:
                out.append(("tool_args", self.call, chunk))

    def _close(self, out: list) -> None:
        if self.state in ("think", "text", "tool"):
            self._emit_text(self._text(b"", final=True), out)
        if self.state == "tool":
            call = self.call
            rest = call.close()
            if rest:
                out.append(("tool_args", call, rest))
            out.append(("tool_end", call))
            self.call = None
        self.header = b""
        self.state = None

    def feed(self, tok: int) -> list:
        out: list = []
        if self.state == "done":
            return out
        if tok in V.STOP_IDS:
            self._close(out)
            self.state = "done"
        elif tok == V.MSG_MODEL:
            self._close(out)
            self.state = "header"
        elif tok == V.END_MESSAGE:
            self._close(out)
        elif tok in self.BODY:
            name = self.header.decode("utf-8", "replace").strip() if self.state == "header" else ""
            if self.state not in (None, "header"):
                self._close(out)
            self.header = b""
            self.state = self.BODY[tok]
            if self.state == "tool":
                kind = "json" if tok == V.CONTENT_TOOL_JSON else "text"
                self.call = ToolCall(len(self.calls), name, kind)
                self.calls.append(self.call)
                out.append(("tool_begin", self.call))
        elif self.vocab.is_special(tok):
            pass                                    # a marker that has no place in a reply: ignored
        elif self.state == "header":
            self.header += self.vocab.token_bytes(tok)
        else:
            if self.state is None:                  # text with no block header: the answer
                self.state = "text"
            self._emit_text(self._text(self.vocab.token_bytes(tok)), out)
        return out

    def finish(self) -> list:
        out: list = []
        if self.state != "done":
            self._close(out)
        return out

    def snapshot(self):
        """(open block, answer text, completed calls, cut splits a character) for prompt.Turn."""
        block = self.state if self.state in ("think", "text", "header", "tool") else None
        calls = [(c.name, c.args) for c in self.calls if c.arguments is not None]
        return block, self.content, calls, self.pending_bytes


class StopMatcher:
    """Finds client stop strings in streamed answer text; holds back a tail that may begin one."""

    def __init__(self, stops):
        if isinstance(stops, str):
            stops = [stops]
        self.stops = [s for s in (stops or []) if isinstance(s, str) and s]
        self.buf = ""
        self.hit = False

    def push(self, text: str) -> str:
        if not self.stops:
            return text
        if self.hit:
            return ""
        self.buf += text
        cut = min((i for i in (self.buf.find(s) for s in self.stops) if i >= 0), default=-1)
        if cut >= 0:
            self.hit = True
            out, self.buf = self.buf[:cut], ""
            return out
        hold = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(self.buf)), 0, -1):
                if self.buf.endswith(s[:k]):
                    hold = max(hold, k)
                    break
        out, self.buf = self.buf[:len(self.buf) - hold], self.buf[len(self.buf) - hold:]
        return out

    def flush(self) -> str:
        out, self.buf = ("" if self.hit else self.buf), ""
        return out
