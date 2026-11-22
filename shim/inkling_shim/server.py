"""OpenAI-compatible HTTP front for strata-inkling: /v1/chat/completions (SSE and plain), /v1/models, /health.

Binds 127.0.0.1 only. Standard library HTTP server, one thread per connection; the engine itself serves one request
at a time (Manager's FIFO lock), the rest wait in arrival order.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import select
import shlex
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .engine import Abandoned, Config, EngineBlocked, EngineError, Manager, MemoryTrip
from .memguard import GuardConfig, MemoryGuard, engine_need_gb
from .prompt import BadRequest, Renderer
from .service import ContextOverflow, Service

HOST = "127.0.0.1"
MAX_BODY = 64 * 1024 * 1024


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def peer_gone(sock) -> bool:
    """True once the client closed its side (readable with nothing to read)."""
    try:
        r, _, _ = select.select([sock], [], [], 0)
        return bool(r) and sock.recv(1, socket.MSG_PEEK) == b""
    except (OSError, ValueError):
        return True


def error_body(message: str, kind: str = "invalid_request_error", code=None, **extra) -> dict:
    return {"error": {"message": message, "type": kind, "code": code, **extra}}


class App:
    def __init__(self, service: Service, alias: str):
        self.service, self.alias = service, alias
        self.started = time.time()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "inkling-s"
    app: App = None  # set by make_server

    def log_message(self, fmt, *args):          # quiet access log; requests are logged by Service
        pass

    # ------------------------------------------------------------ plumbing
    def _json(self, status: int, obj) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> bytes:
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            out = bytearray()
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return bytes(out)
                out += self.rfile.read(size)
                self.rfile.readline()
                if len(out) > MAX_BODY:
                    raise BadRequest("request body too large")
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise BadRequest("request body too large")
        return self.rfile.read(n) if n else b""

    def engine_setup(self) -> dict:
        c = self.app.service.manager.cfg
        g = self.app.service.manager.guard
        return {"engine_cmd": c.engine, "pack": c.pack, "engine_args": c.extra, "draft": c.draft,
                "anti_loop": c.anti_loop, "reuse": self.app.service.reuse, "memory_guard": g is not None,
                "need_gb": g.need_gb if g is not None else None,
                "guard": g.describe() if g is not None else None}

    # ------------------------------------------------------------ routes
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        a = self.app
        m = a.service.manager
        if path == "/health":
            trip = m.trip and {"reason": m.trip[0], "seconds_ago": round(time.monotonic() - m.trip[1])}
            self._json(200, {"status": "ok", "engine": m.state, "queued": m.lock.waiting,
                             "busy": m.lock.held, "ctx": a.service.ctx, "loads": m.loads,
                             "engine_pid": m.engine.pid if m.engine else None,
                             "idle_s": round(time.monotonic() - m.last_used, 1), "last_watchdog_unload": trip,
                             **self.engine_setup(), "stats": a.service.stats})
        elif path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{
                "id": a.alias, "object": "model", "created": int(a.started), "owned_by": "local",
                "context_length": a.service.ctx, "max_model_len": a.service.ctx,
                "meta": {"n_ctx": a.service.ctx, **self.engine_setup()}}]})
        else:
            self._json(404, error_body(f"no route {self.path}", "not_found_error"))

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/unload":
            ok = self.app.service.manager.unload()
            self._json(200 if ok else 409, {"unloaded": ok})
            return
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._json(404, error_body(f"no route {self.path}", "not_found_error"))
            return
        try:
            req = json.loads(self._body() or b"null")
            plan = self.app.service.prepare(req)
        except ContextOverflow as e:
            self._json(400, error_body(str(e), code="context_length_exceeded", n_prompt_tokens=e.n_prompt,
                                       n_ctx=e.n_ctx))
            return
        except (BadRequest, ValueError) as e:
            self._json(400, error_body(str(e)))
            return
        if plan.rendered.literals:
            log(f"note: {plan.rendered.literals} client string(s) contain special-token text; encoded as text")
        self._complete(req, plan)

    # ------------------------------------------------------------ completion
    def _complete(self, req, plan):
        stream = bool(req.get("stream"))
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        cid = "chatcmpl-" + secrets.token_hex(12)
        created = int(time.time())
        alias = self.app.alias
        gone = [False]

        def abandoned():
            if not gone[0] and peer_gone(self.connection):
                gone[0] = True
            return gone[0]

        def send_chunk(delta: dict, finish=None, **extra):
            if gone[0]:
                return
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": alias,
                   "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}], **extra}
            self._sse(obj)

        events = self.app.service.run(plan, abandoned)
        reasoning, content, calls = [], [], {}
        try:
            for ev in events:
                kind = ev[0]
                if kind == "start":
                    if stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.close_connection = True
                        send_chunk({"role": "assistant", "content": None})
                elif kind == "reasoning":
                    reasoning.append(ev[1])
                    stream and send_chunk({"reasoning_content": ev[1]})
                elif kind == "content":
                    content.append(ev[1])
                    stream and send_chunk({"content": ev[1]})
                elif kind == "tool_begin":
                    call = ev[1]
                    call.id = "call_" + secrets.token_hex(12)
                    calls[call.index] = call
                    if stream and call.name:
                        call.announced = True
                        send_chunk({"tool_calls": [{"index": call.index, "id": call.id, "type": "function",
                                                    "function": {"name": call.name, "arguments": ""}}]})
                elif kind in ("tool_args", "tool_end"):
                    call = ev[1]
                    if stream and not getattr(call, "announced", False):
                        call.announced = True
                        send_chunk({"tool_calls": [{"index": call.index, "id": call.id, "type": "function",
                                                    "function": {"name": call.name, "arguments": ""}}]})
                    if kind == "tool_args" and stream:
                        send_chunk({"tool_calls": [{"index": call.index, "function": {"arguments": ev[2]}}]})
                elif kind == "finish":
                    _, finish, usage, timings = ev
                    if stream:
                        send_chunk({}, finish, timings=timings)
                        if include_usage:
                            self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                                       "model": alias, "choices": [], "usage": usage})
                        self._sse_raw(b"data: [DONE]\n\n")
                    else:
                        msg = {"role": "assistant", "content": "".join(content)}
                        if reasoning:
                            msg["reasoning_content"] = "".join(reasoning)
                        done_calls = [c for _, c in sorted(calls.items()) if c.arguments is not None]
                        if done_calls:
                            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                                  "function": {"name": c.name, "arguments": c.arguments}}
                                                 for c in done_calls]
                            if not msg["content"]:
                                msg["content"] = None
                        self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": alias,
                                         "choices": [{"index": 0, "message": msg, "logprobs": None,
                                                      "finish_reason": finish}],
                                         "usage": usage, "timings": timings})
        except Abandoned:
            log("client left while waiting for the engine")
        except EngineBlocked as e:
            log(f"refused to load the engine: {e}")
            self._fail(503, error_body(str(e), "unavailable_error", "model_blocked"), stream)
        except MemoryTrip as e:
            log(f"request failed: {e}")
            self._fail(503, error_body(str(e), "unavailable_error", "memory_watchdog"), stream)
        except ContextOverflow as e:
            self._fail(400, error_body(str(e), code="context_length_exceeded", n_prompt_tokens=e.n_prompt,
                                       n_ctx=e.n_ctx), stream)
        except EngineError as e:
            log(f"engine error: {e}")
            self._fail(500, error_body(str(e), "server_error", "engine_error"), stream)
        finally:
            events.close()

    def _fail(self, status, body, stream):
        if stream and getattr(self, "_sse_started", False):
            try:
                self._sse(body)
                self._sse_raw(b"data: [DONE]\n\n")
            except OSError:
                pass
        else:
            try:
                self._json(status, body)
            except OSError:
                pass

    def _sse(self, obj) -> None:
        self._sse_raw(b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n")

    def _sse_raw(self, data: bytes) -> None:
        self._sse_started = True
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except OSError:
            pass                      # the client left; the next abandoned() check cancels the engine


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(service: Service, alias: str, port: int) -> Server:
    handler = type("BoundHandler", (Handler,), {"app": App(service, alias)})
    return Server((HOST, port), handler)


def env(name, default):
    v = os.environ.get(name)
    return default if v in (None, "") else v


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="OpenAI-compatible shim in front of strata-inkling --serve")
    p.add_argument("--port", type=int, default=int(env("INKLING_S_PORT", 8097)))
    p.add_argument("--model-dir", default=env("INKLING_S_MODEL_DIR", None),
                   help="folder with tokenizer.json and chat_template.jinja (the pack's Hugging Face folder)")
    p.add_argument("--pack", default=env("INKLING_S_PACK", None),
                   help="the .strata pack (default MODEL_DIR/Inkling-2-Mini-SQ-15.strata)")
    p.add_argument("--engine", default=env("INKLING_S_ENGINE", "strata-inkling"),
                   help="engine binary; shell-style words, so a command prefix works (tests: python3 fake.py)")
    p.add_argument("--engine-args", default=env("INKLING_S_ENGINE_ARGS", ""),
                   help="extra engine arguments, shell-style, appended last (e.g. '--ablate /path/DIRS.bin')")
    p.add_argument("--ctx", type=int, default=int(env("INKLING_S_CTX", 65536)))
    p.add_argument("--idle", type=float, default=float(env("INKLING_S_IDLE", 300)))
    p.add_argument("--draft", default=env("INKLING_S_DRAFT", "off"), choices=["off", "auto", "dspark"])
    p.add_argument("--anti-loop", default=env("INKLING_S_ANTI_LOOP", "1"), choices=["0", "1"])
    p.add_argument("--chunk", type=int, default=int(env("INKLING_S_CHUNK", 0)))
    p.add_argument("--effort", default=env("INKLING_S_EFFORT", None),
                   help="default reasoning_effort (none..max or 0..0.99); unset = the template's 0.9")
    p.add_argument("--reuse", default=env("INKLING_S_REUSE", "splice"), choices=["splice", "off"])
    p.add_argument("--literals", default=env("INKLING_S_LITERALS", "text"), choices=["text", "special"])
    p.add_argument("--alias", default=env("INKLING_S_ALIAS", "inkling-2-mini-strata"))
    p.add_argument("--engine-log", default=env("INKLING_S_ENGINE_LOG", str(here / "run" / "engine.log")))
    p.add_argument("--run-dir", default=env("INKLING_S_RUN", str(here / "run")),
                   help="where shim.pid and engine.pid are written")
    p.add_argument("--undeclared-calls", default=env("INKLING_S_UNDECLARED_CALLS", "xml"), choices=["xml", "json"],
                   help="a native tool call in reply to a request without tools: Cline-style XML text, or raw JSON")
    p.add_argument("--debug", action="store_true", default=env("INKLING_S_DEBUG", "0") == "1",
                   help="log every reply's raw text, special tokens included")
    p.add_argument("--need-gb", type=float, default=float(env("INKLING_S_NEED_GB", 0)),
                   help="memory the engine needs, GB (0 = from --ctx and --draft)")
    p.add_argument("--no-guard", action="store_true", default=env("INKLING_S_NO_GUARD", "0") == "1",
                   help="skip the memory guard and watchdog (tests with the fake engine only)")
    g = p.add_argument_group("optional memory-guard checks (all off by default; see inkling_shim/memguard.py)")
    g.add_argument("--guard-lock", default=env("INKLING_S_GUARD_LOCK", None),
                   help="an external lock directory (with owner.json {pid}); no load while a live process holds it")
    g.add_argument("--health-url", default=env("INKLING_S_HEALTH_URL", None),
                   help="a service sharing the machine that must stay healthy; while it is up, no other large model")
    g.add_argument("--pause-file", default=env("INKLING_S_PAUSE_FILE", None),
                   help="if this file exists, the --health-url service counts as paused")
    g.add_argument("--watch-file", default=env("INKLING_S_WATCH_FILE", None),
                   help="a log to watch: a new --watch-pattern line unloads the engine at once")
    g.add_argument("--watch-pattern", default=env("INKLING_S_WATCH_PATTERN", "insufficient memory"))
    g.add_argument("--reserve-gb", type=float, default=float(env("INKLING_S_RESERVE_GB", 8.0)),
                   help="free memory required on top of the engine's need before loading")
    g.add_argument("--floor-pct", type=int, default=int(env("INKLING_S_FLOOR_PCT", 12)),
                   help="unload when free or reclaimable memory drops under this percentage")
    g.add_argument("--warn-grace", type=float, default=float(env("INKLING_S_WARN_GRACE", 0.0)),
                   help="seconds a memory-pressure warning may last before the engine is unloaded (0 = next poll)")
    a = p.parse_args(argv)

    if a.model_dir is None:
        print("set INKLING_S_MODEL_DIR (or --model-dir) to the folder with tokenizer.json and chat_template.jinja",
              file=sys.stderr)
        return 2
    if a.pack is None:
        a.pack = str(Path(a.model_dir) / "Inkling-2-Mini-SQ-15.strata")
    if not Path(a.pack).is_file():
        print(f"pack not found: {a.pack}", file=sys.stderr)
        return 2
    Path(a.engine_log).parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "shim.pid").write_text(f"{os.getpid()}\n")
    cfg = Config(engine=shlex.split(a.engine), pack=a.pack, ctx=a.ctx, draft=a.draft, anti_loop=a.anti_loop == "1",
                 chunk=a.chunk, idle=a.idle, log_path=a.engine_log, pid_path=str(run_dir / "engine.pid"),
                 extra=shlex.split(a.engine_args))
    try:
        cfg.argv()
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    gcfg = GuardConfig(lock_dir=a.guard_lock, health_url=a.health_url, pause_file=a.pause_file,
                       watch_file=a.watch_file, watch_pattern=a.watch_pattern, reserve_gb=a.reserve_gb,
                       floor_pct=a.floor_pct, warn_grace_s=a.warn_grace)
    guard = None if a.no_guard else MemoryGuard(a.need_gb or engine_need_gb(a.ctx, a.draft), gcfg)
    manager = Manager(cfg, guard=guard, log=log)
    renderer = Renderer(a.model_dir, literals=a.literals)
    service = Service(renderer, manager, reuse=a.reuse == "splice", effort=a.effort, log=log, debug=a.debug,
                      undeclared_calls=a.undeclared_calls)
    server = make_server(service, a.alias, a.port)

    def stop(signum, _frame):
        log(f"signal {signum}: shutting down")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log(f"inkling-s listening on http://{HOST}:{a.port}: engine {' '.join(cfg.argv())}")
    log(f"ctx {a.ctx}, idle {a.idle:.0f}s, reuse {a.reuse}, literals {a.literals}, "
        f"effort {a.effort or 'template default'}, memory guard "
        f"{'OFF' if guard is None else 'on (' + guard.describe() + ')'}")
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        manager.shutdown()
        server.server_close()
        (run_dir / "shim.pid").unlink(missing_ok=True)
        log("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
