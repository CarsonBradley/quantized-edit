"""From an OpenAI chat request to the token ids the engine reads.

1. `normalize_messages` turns OpenAI messages into what chat_template.jinja expects (tool-call arguments as objects,
   `developer` as `system`, images replaced by a note: this server has no image input).
2. `Renderer.render` renders chat_template.jinja exactly the way transformers' `apply_chat_template` does (same Jinja
   environment, same `tojson`, same `{% generation %}` tracker) and tokenizes the result. The tracker gives the token
   range of every assistant turn, which step 3 needs.
3. `ReuseMemory.splice` swaps an assistant turn this server generated for the exact ids the model generated (its
   reasoning included), when the client sent that turn back unchanged. The engine keeps its KV cache only when the new
   prompt starts with everything it has already read (ink_serve.h, all-or-nothing prefix reuse), and clients such as
   Cline drop the reasoning from the history, so without this every turn would read the whole prompt again.

Special-token text inside a client's strings (a file that contains "<|end_message|>", say) is tokenized as plain
text by default, as Strata Control does; only the tokens the template itself writes are special. With
`literals="special"` it becomes special tokens, as transformers and llama.cpp do.
"""
from __future__ import annotations

import hashlib
import json
import re
from array import array
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import jinja2
from jinja2.ext import Extension, loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from . import vocab as V

OPEN, CLOSE = "﷐", "﷑"          # marks a client string while rendering (Unicode noncharacters)
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
IMAGE_NOTE = "[image omitted: this server (inkling-s) has no image input]"


class BadRequest(ValueError):
    """A request this server answers with HTTP 400."""


# ---------------------------------------------------------------- request normalisation

def _norm_call(tc) -> dict:
    if not isinstance(tc, dict) or not isinstance(tc.get("function"), dict):
        raise BadRequest("each tool_calls entry needs a function object")
    fn = tc["function"]
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        raise BadRequest("tool call function name must be a non-empty string")
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            args = {"raw": args}
    if args is None:
        args = {}
    if not isinstance(args, dict):
        args = {"value": args}
    return {"id": tc.get("id") if isinstance(tc.get("id"), str) else "", "type": "function",
            "function": {"name": name, "arguments": args}}


def normalize_messages(messages) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise BadRequest("messages must be a non-empty list")
    out = []
    for m in messages:
        if not isinstance(m, dict):
            raise BadRequest("each message must be an object")
        role = "system" if m.get("role") == "developer" else m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise BadRequest(f"unsupported message role {m.get('role')!r}")
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, str):
                    parts.append({"type": "text", "text": p})
                elif isinstance(p, dict) and p.get("type") in (None, "text", "input_text"):
                    parts.append({"type": "text", "text": p.get("text") if isinstance(p.get("text"), str) else ""})
                elif isinstance(p, dict) and p.get("type") in ("image", "input_image", "image_url"):
                    parts.append({"type": "text", "text": IMAGE_NOTE})
                else:
                    raise BadRequest(f"unsupported content part {p.get('type') if isinstance(p, dict) else p!r}")
            content = parts
        elif content is not None and not isinstance(content, str):
            raise BadRequest("message content must be a string, a list of parts or null")
        msg = {"role": role, "content": content}
        for k in ("name", "tool_call_id"):
            if isinstance(m.get(k), str) and m[k]:
                msg[k] = m[k]
        if role == "assistant":
            r = m.get("reasoning_content")
            if not isinstance(r, str):
                r = m.get("reasoning") if isinstance(m.get("reasoning"), str) else None
            if r:
                msg["reasoning_content"] = r
            if m.get("tool_calls"):
                if not isinstance(m["tool_calls"], list):
                    raise BadRequest("tool_calls must be a list")
                msg["tool_calls"] = [_norm_call(tc) for tc in m["tool_calls"]]
        out.append(msg)
    return out


def normalize_tools(tools):
    if tools is None or tools == []:
        return None
    if not isinstance(tools, list):
        raise BadRequest("tools must be a list")
    for t in tools:
        fn = t.get("function", t) if isinstance(t, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise BadRequest("each tool needs a function with a name")
    return tools


def resolve_effort(req: dict, default=None):
    """The template's reasoning_effort: chat_template_kwargs first, then reasoning_effort, then reasoning.effort.
    None leaves the template's own default (0.9, thinking on), which is what llama.cpp renders."""
    ctk = req.get("chat_template_kwargs") if isinstance(req.get("chat_template_kwargs"), dict) else {}
    if ctk.get("enable_thinking") is False:
        return "none"
    v = ctk.get("reasoning_effort", req.get("reasoning_effort"))
    if v is None and isinstance(req.get("reasoning"), dict):
        v = req["reasoning"].get("effort")
    if v is None or v == "":
        v = default
    if v is None or v == "":
        return None
    if isinstance(v, str) and v.strip().lower() in EFFORTS:
        return v.strip().lower()
    try:
        num = float(v)
    except (TypeError, ValueError):
        raise BadRequest(f"reasoning_effort must be one of {', '.join(EFFORTS)} or a number in [0, 0.99]")
    if not 0.0 <= num <= 0.99:
        raise BadRequest("reasoning_effort must be in [0.0, 0.99]")
    return num


def message_signature(msg: dict):
    """(content text, [(tool name, args)]) of an assistant message, the way a client sends it back."""
    c = msg.get("content")
    if isinstance(c, list):
        c = "".join(p if isinstance(p, str) else (p.get("text") or "") for p in c)
    calls = [(tc["function"]["name"], tc["function"]["arguments"]) for tc in msg.get("tool_calls") or []]
    return c or "", calls


# ---------------------------------------------------------------- template rendering

class _GenerationTracker(Extension):
    """transformers' AssistantTracker: `{% generation %}` renders its body and records its character span."""
    tags = {"generation"}

    def __init__(self, environment):
        super().__init__(environment)
        environment.extend(activate_tracker=self.activate_tracker)
        self._blocks = None
        self._spans = None

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_generation_support"), [], [], body).set_lineno(lineno)

    @jinja2.pass_eval_context
    def _generation_support(self, context, caller):
        rv = caller()
        if self._blocks is not None:
            start = len("".join(self._blocks))
            self._spans.append((start, start + len(rv)))
        return rv

    @contextmanager
    def activate_tracker(self, blocks, spans):
        self._blocks, self._spans = blocks, spans
        try:
            yield
        finally:
            self._blocks = self._spans = None


def _raise(message):
    raise jinja2.exceptions.TemplateError(message)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def make_environment() -> ImmutableSandboxedEnvironment:
    """The same environment as transformers' _compile_jinja_template (5.x)."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[_GenerationTracker, loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env


@dataclass
class Rendered:
    text: str                       # the template's output, character for character
    ids: list[int]
    spans: list[tuple[int, int]] | None   # token range of each assistant turn (None: could not be located)
    assistants: list[dict]          # the assistant messages, in the same order as spans
    literals: int = 0               # client strings that contained special-token text


class Renderer:
    def __init__(self, model_dir: str | Path, vocab: V.Vocab | None = None, literals: str = "text"):
        self.vocab = vocab or V.Vocab(model_dir)
        self.source = (Path(model_dir) / "chat_template.jinja").read_text(encoding="utf-8")
        self.template = make_environment().from_string(self.source)
        if literals not in ("text", "special"):
            raise ValueError("literals must be 'text' or 'special'")
        self.literals = literals
        specials = sorted(self.vocab.special, key=len, reverse=True)
        self._has_special = re.compile("|".join(re.escape(s) for s in specials))
        self._scan = re.compile("|".join([re.escape(s) for s in specials] + [OPEN, CLOSE]))

    def _protect(self, obj, count):
        if isinstance(obj, str):
            if self._has_special.search(obj) or OPEN in obj or CLOSE in obj:
                count[0] += 1
                return OPEN + obj.replace(OPEN, "").replace(CLOSE, "") + CLOSE
            return obj
        if isinstance(obj, list):
            return [self._protect(x, count) for x in obj]
        if isinstance(obj, dict):
            return {k: (v if k in ("role", "type") else self._protect(v, count)) for k, v in obj.items()}
        return obj

    def render_text(self, messages, tools=None, effort=None, track=None):
        kwargs = {"messages": messages, "tools": tools, "documents": None, "add_generation_prompt": False}
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        blocks, spans = [], []
        with self.template.environment.activate_tracker(blocks, spans):
            for block in self.template.generate(**kwargs):
                blocks.append(block)
        if track is not None:
            track.extend(spans)
        return "".join(blocks)

    def render(self, messages: list[dict], tools=None, effort=None) -> Rendered:
        count = [0]
        if self.literals == "text":
            messages, tools = self._protect(messages, count), self._protect(tools, count)
        char_spans: list[tuple[int, int]] = []
        try:
            raw = self.render_text(messages, tools, effort, char_spans)
        except jinja2.exceptions.TemplateError as e:
            raise BadRequest(f"chat template: {e}") from None
        ids, spans = self._encode(raw, char_spans)
        assistants = [self._unprotect(m) for m in messages if m["role"] == "assistant"]
        if spans is not None and len(spans) != len(assistants):
            spans = None
        return Rendered(raw.replace(OPEN, "").replace(CLOSE, ""), ids, spans, assistants, count[0])

    def _unprotect(self, obj):
        if isinstance(obj, str):
            return obj.replace(OPEN, "").replace(CLOSE, "")
        if isinstance(obj, list):
            return [self._unprotect(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._unprotect(v) for k, v in obj.items()}
        return obj

    def _encode(self, raw: str, char_spans):
        """Tokenize: special-token text the template wrote -> its id; everything between -> plain text."""
        starts = {a: i for i, (a, _) in enumerate(char_spans)}
        ends = {b: i for i, (_, b) in enumerate(char_spans)}
        tok_spans = [[None, None] for _ in char_spans]
        ids: list[int] = []
        text: list[str] = []
        depth, last = 0, 0
        for m in self._scan.finditer(raw):
            text.append(raw[last:m.start()])
            last = m.end()
            t = m.group(0)
            if t == OPEN:
                depth += 1
            elif t == CLOSE:
                depth -= 1
            elif depth > 0:
                text.append(t)
            else:
                if text:
                    ids.extend(self.vocab.encode_text("".join(text)))
                    text.clear()
                if m.start() in starts:
                    tok_spans[starts[m.start()]][0] = len(ids)
                ids.append(self.vocab.special[t])
                if m.end() in ends:
                    tok_spans[ends[m.end()]][1] = len(ids)
        text.append(raw[last:])
        ids.extend(self.vocab.encode_text("".join(text)))
        if any(a is None or b is None for a, b in tok_spans):
            return ids, None
        return ids, [(a, b) for a, b in tok_spans]


# ---------------------------------------------------------------- prefix reuse across turns

def digest(ids) -> str:
    return hashlib.sha256(array("i", ids).tobytes()).hexdigest()


@dataclass
class Turn:
    """One generation as the engine holds it: every generated id it fed back (all but the last one sampled)."""
    fed: list[int]
    complete: bool                  # the model ended its turn itself
    open_block: str | None          # the block the cut falls in: None, "think", "text", "header", "tool"
    content: str                    # answer text inside `fed`
    tool_calls: list = field(default_factory=list)   # [(name, args)] completed inside `fed`
    clean: bool = True              # False: the cut splits a UTF-8 character inside the answer

    def span_for(self, msg: dict, vocab: V.Vocab):
        """The ids that stand for `msg` (the client's copy of this turn) and start with `fed`, or None."""
        c, calls = message_signature(msg)
        if calls != self.tool_calls or not self.clean:
            return None
        if self.complete and c.strip() == self.content.strip():
            return self.fed + [V.END_SAMPLING]
        if not c.startswith(self.content):
            return None
        rest = c[len(self.content):]
        if self.tool_calls and rest.strip():
            return None                 # the template would put that text before the calls
        tail: list[int] = []
        if self.open_block == "text":
            tail += vocab.encode_text(rest) + [V.END_MESSAGE]
            rest = ""
        elif self.open_block == "think":
            tail.append(V.END_MESSAGE)
        elif self.open_block is not None:
            return None
        if rest.strip():
            tail += [V.MSG_MODEL, V.CONTENT_TEXT] + vocab.encode_text(rest) + [V.END_MESSAGE]
        return self.fed + tail + [V.END_SAMPLING]


class ReuseMemory:
    """prompt digest -> the Turn generated for it. Bounded; lives as long as the server process."""

    def __init__(self, vocab: V.Vocab, capacity: int = 64):
        self.vocab = vocab
        self.capacity = capacity
        self.turns: OrderedDict[str, Turn] = OrderedDict()

    def remember(self, prompt_ids, turn: Turn) -> None:
        key = digest(prompt_ids)
        self.turns[key] = turn
        self.turns.move_to_end(key)
        while len(self.turns) > self.capacity:
            self.turns.popitem(last=False)

    def splice(self, r: Rendered) -> tuple[list[int], int]:
        """The prompt with every remembered assistant turn replaced by its generated ids -> (ids, turns replaced)."""
        if not r.spans or not self.turns:
            return list(r.ids), 0
        out: list[int] = []
        h = hashlib.sha256()
        prev, n = 0, 0
        for (a, b), msg in zip(r.spans, r.assistants):
            seg = r.ids[prev:a]
            out += seg
            h.update(array("i", seg).tobytes())
            turn = self.turns.get(h.copy().hexdigest())
            span = turn.span_for(msg, self.vocab) if turn else None
            if span is None:
                span = r.ids[a:b]
            else:
                n += 1
            out += span
            h.update(array("i", span).tobytes())
            prev = b
        out += r.ids[prev:]
        return out, n
