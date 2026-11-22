"""One chat completion, independent of HTTP: request -> prompt ids -> engine -> parsed events.

`Service.prepare` validates and renders (CPU only, before waiting for the engine); `Service.run` waits for the
engine, sends the prompt and yields events:
    ("start", info) ("reasoning", text) ("content", text)
    ("tool_begin", ToolCall) ("tool_args", ToolCall, text) ("tool_end", ToolCall)
    ("finish", finish_reason, usage, timings)
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field

from . import vocab as V
from .engine import EngineError, Manager, MemoryTrip, Run
from .parse import OutputParser, StopMatcher
from .prompt import (BadRequest, Renderer, Rendered, ReuseMemory, Turn, normalize_messages, normalize_tools,
                     resolve_effort)

# What a request leaves unset gets llama.cpp server's defaults, so a client sees the same sampling as on llama-server.
LLAMA_DEFAULTS = {"temperature": 0.8, "top_p": 0.95, "top_k": 40, "min_p": 0.05}


def call_as_text(call, mode: str, before: str = "") -> str:
    """A native tool call the request did not ask for (no `tools`), as answer text. Cline drives models with its
    own XML tool format in plain text, and this model sometimes answers such prompts with a native call instead;
    "xml" renders it the way Cline expects (<name><param>value</param>...</name>), "json" passes the raw block."""
    if mode != "xml" or not isinstance(call.args, dict) or set(call.args) <= {"raw", "input"} and call.args:
        return call.body
    lines = [f"<{call.name}>"]
    for k, v in call.args.items():
        lines.append(f"<{k}>{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}</{k}>")
    lines.append(f"</{call.name}>")
    sep = "" if not before or before.endswith("\n\n") else ("\n" if before.endswith("\n") else "\n\n")
    return sep + "\n".join(lines)


class ContextOverflow(BadRequest):
    def __init__(self, n_prompt: int, n_ctx: int):
        super().__init__(f"This model's maximum context length is {n_ctx} tokens. However, your messages resulted "
                         f"in {n_prompt} tokens (the request exceeds the available context size).")
        self.n_prompt, self.n_ctx = n_prompt, n_ctx


@dataclass
class Plan:
    rendered: Rendered
    tools: list | None
    max_tokens: int | None
    sampling: dict
    stop: list = field(default_factory=list)
    tools_enabled: bool = True


def _num(req, key, lo, hi, cast=float):
    v = req.get(key)
    if v is None:
        return None
    try:
        v = cast(v)
    except (TypeError, ValueError):
        raise BadRequest(f"{key} must be a number")
    if not lo <= v <= hi:
        raise BadRequest(f"{key} must be in [{lo}, {hi}]")
    return v


class Service:
    def __init__(self, renderer: Renderer, manager: Manager, *, reuse: bool = True, effort=None, log=print,
                 debug: bool = False, undeclared_calls: str = "xml"):
        self.renderer, self.manager = renderer, manager
        self.vocab = renderer.vocab
        self.memory = ReuseMemory(self.vocab)
        self.reuse = reuse
        self.effort = effort
        self.log = log
        self.undeclared_calls = undeclared_calls   # how a native call is shown when the request declared no tools
        self.debug = debug                 # log every reply's raw text, special tokens included
        self.stats = {"requests": 0, "prompt_tokens": 0, "reused_tokens": 0, "reread_tokens": 0,
                      "generated_tokens": 0}

    @property
    def ctx(self) -> int:
        return self.manager.cfg.ctx

    def prepare(self, req: dict) -> Plan:
        if not isinstance(req, dict):
            raise BadRequest("body must be a JSON object")
        if req.get("n") not in (None, 1):
            raise BadRequest("n > 1 is not supported")
        messages = normalize_messages(req.get("messages"))
        tools = normalize_tools(req.get("tools"))
        effort = resolve_effort(req, self.effort)
        rendered = self.renderer.render(messages, tools, effort)
        if len(rendered.ids) >= self.ctx:
            raise ContextOverflow(len(rendered.ids), self.ctx)
        max_tokens = req.get("max_completion_tokens", req.get("max_tokens"))
        if max_tokens is not None:
            max_tokens = int(_num(req, "max_completion_tokens" if "max_completion_tokens" in req else "max_tokens",
                                  1, 10 ** 9, int))
        s = dict(LLAMA_DEFAULTS)
        for key, lo, hi in (("temperature", 0, 5), ("top_p", 0, 1), ("top_k", 0, 10 ** 6), ("min_p", 0, 1)):
            v = _num(req, key, lo, hi)
            if v is not None:
                s[key] = v
        if s["top_k"] <= 0:
            s["top_k"] = 128                           # llama.cpp: 0 = off; the engine never looks past 128
        s["top_k"] = int(min(128, s["top_k"]))
        for key, engine_key in (("presence_penalty", "penalty_present"), ("frequency_penalty", "penalty_freq"),
                                ("repeat_penalty", "penalty_repeat")):
            v = _num(req, key, -2 if key != "repeat_penalty" else 0, 2 if key != "repeat_penalty" else 10)
            if v is not None:
                s[engine_key] = v
        seed = _num(req, "seed", -2 ** 63, 2 ** 64 - 1, int)
        s["seed"] = seed if seed is not None and seed > 0 else random.randrange(1, 2 ** 63)
        stop = req.get("stop") or []
        stop = [stop] if isinstance(stop, str) else stop
        if not isinstance(stop, list) or not all(isinstance(x, str) for x in stop):
            raise BadRequest("stop must be a string or a list of strings")
        enabled = bool(tools) and req.get("tool_choice") != "none"
        return Plan(rendered, tools, max_tokens, s, stop, enabled)

    def _prompt_ids(self, r: Rendered, eng) -> tuple[list[int], int]:
        ids, n = (self.memory.splice(r) if self.reuse else (list(r.ids), 0))
        if n and len(ids) >= eng.ctx:
            self.log(f"spliced prompt ({len(ids)} tokens) does not fit {eng.ctx}; using the plain rendering")
            ids, n = list(r.ids), 0
        if len(ids) >= eng.ctx:
            raise ContextOverflow(len(ids), eng.ctx)
        return ids, n

    def run(self, plan: Plan, abandoned=lambda: False):
        with self.manager.session(abandoned) as eng:
            ids, spliced = self._prompt_ids(plan.rendered, eng)
            fed = eng.fed
            expect = len(fed) if fed and len(fed) < len(ids) and ids[:len(fed)] == fed else 0
            miss = self._miss(fed, ids) if fed and not expect else ""
            max_new = eng.ctx - len(ids)
            if plan.max_tokens:
                max_new = min(max_new, plan.max_tokens)
            run = Run(eng, ids, max_new, plan.sampling)
            t0 = time.monotonic()
            yield ("start", {"prompt_tokens": len(ids), "spliced_turns": spliced, "expected_reuse": expect})
            parser = OutputParser(self.vocab)
            stopper = StopMatcher(plan.stop)
            sent = ""
            why_cancel = None
            t_first = None
            try:
                for ev in run.events():
                    if why_cancel is None and (abandoned() or self.manager.closing):
                        why_cancel = "client gone" if not self.manager.closing else "shutdown"
                        run.cancel()
                    if ev[0] != "token" or why_cancel:
                        continue
                    if t_first is None:
                        t_first = time.monotonic()
                    for pev in parser.feed(ev[1]):
                        for out in self._relay(pev, stopper, plan, sent):
                            if out[0] == "content":
                                sent += out[1]
                            yield out
                    if stopper.hit:
                        why_cancel = "stop string"
                        run.cancel()
            except EngineError as e:
                if eng.tripped:
                    raise MemoryTrip(f"the memory watchdog unloaded the model mid-reply: {eng.tripped}") from e
                raise
            finally:
                if run.done is None and eng.alive():   # stopped early (an error, or the generator closed)
                    run.cancel()
                    try:
                        for _ in run.events():
                            pass
                    except EngineError as e:
                        self.log(f"while cancelling: {e}")
            if why_cancel in ("client gone", "shutdown"):
                self._finish_log(run, ids, spliced, expect, t0, t_first, why_cancel, miss)
                self._remember(ids, run, plan)
                if why_cancel == "shutdown":
                    raise EngineError("the server is shutting down; this reply was cancelled")
                return
            if not stopper.hit:
                for pev in parser.finish():
                    yield from self._relay(pev, stopper, plan, sent)
                tail = stopper.flush()
                if tail:
                    yield ("content", tail)
            self._remember(ids, run, plan)
            done = run.done
            calls = [c for c in parser.calls if c.arguments is not None]
            if stopper.hit:
                finish = "stop"
            elif done.reason == "length":
                finish = "length"
            elif calls and plan.tools_enabled:
                finish = "tool_calls"
            else:
                finish = "stop"
            if done.reason == "stop" and run.gen and run.gen[-1] not in V.STOP_IDS:
                self.log("note: the engine's anti-loop check ended this reply (repeated tokens)")
            self._finish_log(run, ids, spliced, expect, t0, t_first, finish, miss)
            n_new = len(ids) - run.resume
            usage = {"prompt_tokens": len(ids), "completion_tokens": len(run.gen),
                     "total_tokens": len(ids) + len(run.gen),
                     "prompt_tokens_details": {"cached_tokens": run.resume}}
            timings = {"cache_n": run.resume, "prompt_n": n_new, "prompt_ms": done.prompt_ms,
                       "prompt_per_second": n_new / (done.prompt_ms / 1000) if done.prompt_ms > 0 else None,
                       "predicted_n": len(run.gen), "predicted_ms": done.decode_ms,
                       "predicted_per_second": len(run.gen) / (done.decode_ms / 1000) if done.decode_ms > 0 else None,
                       "engine_reason": done.reason, "spliced_turns": spliced}
            yield ("finish", finish, usage, timings)

    def _relay(self, pev, stopper: StopMatcher, plan: Plan, sent: str = ""):
        kind = pev[0]
        if kind == "content":
            text = stopper.push(pev[1])
            if text:
                yield ("content", text)
        elif kind == "reasoning":
            yield pev
        elif plan.tools_enabled:
            yield pev
        elif kind == "tool_end":                     # no tools in the request: the call is passed on as text
            text = stopper.push(call_as_text(pev[1], self.undeclared_calls, sent))
            if text:
                yield ("content", text)

    def _remember(self, ids, run: Run, plan: Plan) -> None:
        """Store what the engine now holds and what the client saw of it (prompt.Turn), for splicing."""
        if not run.gen:
            return
        fed = run.gen[:-1]
        p = OutputParser(self.vocab)
        content, calls, none = "", [], StopMatcher(None)
        for t in fed:
            for pev in p.feed(t):
                for out in self._relay(pev, none, plan, content):
                    if out[0] == "content":
                        content += out[1]
                    elif out[0] == "tool_end":
                        calls.append((out[1].name, out[1].args))
        block, _, _, pending = p.snapshot()
        complete = run.gen[-1] in V.STOP_IDS
        self.memory.remember(ids, Turn(list(fed), complete, block, content, calls,
                                       clean=not (block == "text" and pending)))

    def _miss(self, fed, ids) -> str:
        """Why the engine's cache will not be reused: where the new prompt stops matching it."""
        n = min(len(fed), len(ids))
        i = next((k for k in range(n) if fed[k] != ids[k]), n)
        if i == len(fed):
            return f"cache miss: the prompt ({len(ids)}) is not longer than the cached {len(fed)} tokens"
        if i == len(ids):
            return (f"cache miss: the prompt is a prefix of the engine's {len(fed)} cached tokens (a repeated "
                    f"request?); the engine can only extend its cache")
        return (f"cache miss: the prompt differs from the engine's {len(fed)} cached tokens at token {i}: cached "
                f"{self.vocab.decode(fed[max(0, i - 8):i + 8])!r} vs new {self.vocab.decode(ids[max(0, i - 8):i + 8])!r}")

    def _finish_log(self, run: Run, ids, spliced, expect, t0, t_first, finish, miss="") -> None:
        d = run.done
        self.stats["requests"] += 1
        self.stats["prompt_tokens"] += len(ids)
        self.stats["reused_tokens"] += run.resume
        self.stats["generated_tokens"] += len(run.gen)
        reread = (len(ids) if run.gen else max(run.pp, run.resume)) - run.resume   # cancelled mid-prompt: what it read
        self.stats["reread_tokens"] += reread
        pre = f"re-read {reread} in {d.prompt_ms / 1000:.2f}s" if d else "no DONE"
        rate = f" ({reread / (d.prompt_ms / 1000):.0f} tok/s)" if d and d.prompt_ms > 0 else ""
        dec = f"decode {len(run.gen)} tok in {d.decode_ms / 1000:.2f}s" if d else ""
        drate = f" ({len(run.gen) / (d.decode_ms / 1000):.1f} tok/s)" if d and d.decode_ms > 0 else ""
        warn = ""
        if run.resume < expect:
            warn = f"; REUSE MISSED: expected {expect}, the engine reused {run.resume}"
        elif miss:
            warn = "; " + miss
        cancel = f"; cancel latency {run.cancel_ms:.0f} ms" if run.cancel_ms is not None else ""
        self.log(f"{run.rid}: prompt {len(ids)} tok: reused {run.resume}, {pre}{rate}; spliced turns {spliced}; "
                 f"{dec}{drate}; engine {d.reason if d else '-'}; finish {finish}{cancel}; "
                 f"wall {time.monotonic() - t0:.1f}s{warn}")
        if self.debug and run.gen:
            self.log(f"{run.rid} raw output: {self.vocab.decode(run.gen)[:4000]!r}")
