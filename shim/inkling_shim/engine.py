"""The `strata-inkling --serve` child process and its lifecycle.

Engine: one child process, its stdout read by a thread into a queue, its stderr appended to a log file.
Run: one GENERATE request on it (line protocol in docs/serve-protocol.md; Strata source ink_serve.h).
Manager: at most one engine; requests take turns in arrival order (FifoLock); the engine is started on the first
request (after the memory guard agrees), and stopped after `idle` seconds without a request or on shutdown.
"""
from __future__ import annotations

import collections
import os
import queue
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


class EngineError(RuntimeError):
    pass


class EngineBlocked(EngineError):
    """The guard refused to load the model (another large model is in memory)."""


class MemoryTrip(EngineError):
    """The memory watchdog unloaded the engine (a watched log, memory pressure or low free memory)."""


class Abandoned(Exception):
    """The client went away while waiting for its turn."""


@dataclass
class Config:
    engine: list[str]                    # argv prefix: [path to strata-inkling] (tests: [python, fake_engine.py])
    pack: str
    ctx: int = 65536
    draft: str = "off"                   # auto | dspark | off
    anti_loop: bool = True
    chunk: int = 0                       # 0: the engine's default (512)
    idle: float = 300.0                  # seconds without a request before the engine is stopped; <= 0: never
    log_path: str | None = None          # engine stderr
    pid_path: str | None = None          # the running engine's pid, for `inkling-s stop` to double-check
    ready_timeout: float = 600.0
    silence_timeout: float = 300.0       # a running request whose engine prints nothing this long: engine killed
    extra: list[str] = field(default_factory=list)   # INKLING_S_ENGINE_ARGS, e.g. ["--ablate", "DIRS.bin"]

    # One slot only: the shim relies on slot 0's KV cache, and the stock 0.1.65 engine returns token 0 as the first
    # token of a GENERATE admitted next to another one when it has several slots (docs/upstream-bugs.md). In ink_main.cpp
    # --batch and --slots are aliases of --parallel.
    SLOT_FLAGS = ("--parallel", "--slots", "--batch")

    def argv(self) -> list[str]:
        bad = [x for x in self.extra if x in self.SLOT_FLAGS or x.split("=")[0] in self.SLOT_FLAGS]
        if bad:
            raise ValueError(f"engine args must not set the slot count ({', '.join(bad)}): the shim runs one slot")
        a = [*self.engine, "--serve", "--native", self.pack, "--max-context", str(self.ctx), "--parallel", "1",
             "--draft", self.draft]
        if not self.anti_loop:
            a.append("--no-anti-loop")
        if self.chunk:
            a += ["--chunk", str(self.chunk)]
        return a + list(self.extra)


class Engine:
    def __init__(self, cfg: Config, log=print):
        self.cfg = cfg
        self.log = log
        self.lines: queue.Queue = queue.Queue()
        self.fed: list[int] = []         # what the engine's slot 0 holds in its KV (mirror of cached_prefix)
        self.info = ""
        self.ctx = cfg.ctx
        self.tripped: str | None = None  # set when the memory watchdog killed it
        self._rid = 0
        self._errf = open(cfg.log_path, "ab") if cfg.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(cfg.argv(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._errf,
                                     text=True, bufsize=1, start_new_session=True)
        threading.Thread(target=self._read, name="engine-stdout", daemon=True).start()

    def _read(self):
        try:
            for line in self.proc.stdout:
                self.lines.put(line.rstrip("\r\n"))
        except (OSError, ValueError):
            pass
        self.lines.put(None)

    @property
    def pid(self) -> int:
        return self.proc.pid

    def alive(self) -> bool:
        return self.proc.poll() is None

    def wait_ready(self, timeout: float) -> None:
        end = time.monotonic() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.05, end - time.monotonic()))
            except queue.Empty:
                self.kill()
                raise EngineError(f"engine not ready after {timeout:.0f}s")
            if line is None:
                self.kill()
                raise EngineError(f"engine exited while loading (code {self.proc.poll()}); see the engine log")
            if line.startswith("INFO"):
                self.info = line[5:]
            elif line.startswith("READY"):
                parts = line.split()
                if len(parts) > 1 and parts[1].isdigit():
                    self.ctx = int(parts[1])
                return
            elif line.startswith("ERR"):
                self.kill()
                raise EngineError(line)

    def send(self, line: str) -> None:
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            raise EngineError(f"engine stdin closed: {e}") from None

    def next_line(self, timeout: float):
        return self.lines.get(timeout=timeout)    # raises queue.Empty; None = the engine exited

    def new_rid(self) -> str:
        self._rid += 1
        return f"r{self._rid}"

    def quit(self, timeout: float = 20.0) -> None:
        """QUIT, then TERM, then KILL. The engine finishes a running request before it reads QUIT, so callers
        cancel first."""
        if self.alive():
            try:
                self.send("QUIT")
                self.proc.stdin.close()
            except (EngineError, OSError):
                pass
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.kill()
        self._close_log()

    def kill(self) -> None:
        if self.alive():
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.fed = []
        self._close_log()

    def _close_log(self):
        if self._errf not in (None, subprocess.DEVNULL):
            try:
                self._errf.close()
            except OSError:
                pass


@dataclass
class Done:
    generated: int
    prompt_len: int
    prompt_ms: float
    decode_ms: float
    reason: str                          # stop | length | cancel | nonfinite


class Run:
    """One GENERATE. Iterate `events()`; call `cancel()` to stop it (the engine answers with DONE ... cancel)."""

    def __init__(self, engine: Engine, ids: list[int], max_new: int, sampling: dict):
        self.engine, self.ids = engine, ids
        self.rid = engine.new_rid()
        self.gen: list[int] = []
        self.pp = 0
        self.resume = 0
        self.done: Done | None = None
        self.cancelled = False
        self.t_cancel = None
        self.cancel_ms = None             # CANCEL sent -> DONE received
        self.t0 = time.monotonic()
        keys = "".join(f" {k}={v}" for k, v in sampling.items())
        engine.send(f"GENERATE {self.rid} {max_new}{keys} {','.join(map(str, ids))}")

    def cancel(self) -> None:
        if not self.cancelled and self.done is None:
            self.cancelled = True
            self.t_cancel = time.monotonic()
            try:
                self.engine.send(f"CANCEL {self.rid}")
            except EngineError:
                pass

    def events(self, poll: float = 0.25):
        """Yields ("tick",), ("resume", n), ("prefill", done, total), ("token", id); returns after DONE."""
        e = self.engine
        quiet_since = time.monotonic()
        while self.done is None:
            try:
                line = e.next_line(poll)
            except queue.Empty:
                if not e.alive() and e.lines.empty():
                    e.fed = []
                    raise EngineError(f"engine exited (code {e.proc.poll()}); see the engine log")
                if time.monotonic() - quiet_since > e.cfg.silence_timeout:
                    e.kill()
                    raise EngineError(f"engine silent for {e.cfg.silence_timeout:.0f}s; stopped it")
                yield ("tick",)
                continue
            quiet_since = time.monotonic()
            if line is None:
                try:                      # reap it, so alive() is False before the next request looks
                    e.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
                e.kill()
                raise EngineError(f"engine exited (code {e.proc.poll()}); see the engine log")
            head, _, rest = line.partition(" ")
            if head == "T":
                rid, _, tok = rest.partition(" ")
                if rid == self.rid:
                    self.gen.append(int(tok))
                    yield ("token", int(tok))
            elif head == "PP":
                p = rest.split()
                self.pp = int(p[0])
                yield ("prefill", int(p[0]), int(p[1]))
            elif head == "RESUME":
                self.resume = int(rest.split()[0])
                yield ("resume", self.resume)
            elif head == "DONE":
                p = rest.split()
                if p and p[0] == self.rid and len(p) >= 6:
                    self.done = Done(int(p[1]), int(p[2]), float(p[3]), float(p[4]), p[5])
                    if self.t_cancel is not None:
                        self.cancel_ms = (time.monotonic() - self.t_cancel) * 1000
                    self._update_fed()
            elif head == "ERR":
                if "fatal" in rest:
                    e.kill()
                raise EngineError(line)
            # INFO / HELLO / STATS / anything else: not ours

    def _update_fed(self) -> None:
        if self.gen:                      # prefill finished; the last sampled token was never fed back
            self.engine.fed = self.ids + self.gen[:-1]
        else:
            self.engine.fed = self.ids[:max(self.pp, self.resume)]


class FifoLock:
    """A lock granted in arrival order; waiters can give up (client gone) while they wait."""

    def __init__(self):
        self._cv = threading.Condition()
        self._queue: collections.deque = collections.deque()
        self._held = False

    def acquire(self, abandoned=lambda: False, poll: float = 0.5) -> bool:
        me = object()
        with self._cv:
            self._queue.append(me)
            try:
                while self._held or self._queue[0] is not me:
                    if abandoned():
                        return False
                    self._cv.wait(poll)
                self._held = True
                return True
            finally:
                self._queue.remove(me)
                self._cv.notify_all()

    def try_acquire(self) -> bool:
        with self._cv:
            if self._held or self._queue:
                return False
            self._held = True
            return True

    def release(self) -> None:
        with self._cv:
            self._held = False
            self._cv.notify_all()

    @property
    def waiting(self) -> int:
        with self._cv:
            return len(self._queue)

    @property
    def held(self) -> bool:
        return self._held


class Manager:
    """guard: a memguard.MemoryGuard (preflight / arm / check / poll_s / cfg.cooldown_s), or None for no checks."""

    def __init__(self, cfg: Config, guard=None, log=print):
        self.cfg, self.guard, self.log = cfg, guard, log
        self.engine: Engine | None = None
        self._loading: Engine | None = None
        self.trip: tuple[str, float] | None = None   # (reason, monotonic time) of the last watchdog unload
        self.lock = FifoLock()
        self.last_used = time.monotonic()
        self.state = "unloaded"           # unloaded | loading | loaded
        self.closing = False
        self.loads = 0
        self._stop = threading.Event()
        self._idle = threading.Thread(target=self._idle_loop, name="idle-unload", daemon=True)
        self._idle.start()
        if guard is not None:
            threading.Thread(target=self._watch_loop, name="memory-watchdog", daemon=True).start()

    @contextmanager
    def session(self, abandoned=lambda: False):
        """Wait for this request's turn, make sure the engine runs, yield it."""
        if not self.lock.acquire(lambda: abandoned() or self.closing):
            raise Abandoned()
        try:
            if self.closing:
                raise EngineError("server is shutting down")
            yield self._ensure()
        finally:
            self.last_used = time.monotonic()
            self.lock.release()

    def _ensure(self) -> Engine:
        if self.engine is not None and self.engine.alive():
            return self.engine
        if self.engine is not None:
            self.log(f"engine pid {self.engine.pid} had exited (code {self.engine.proc.poll()}); starting it again")
            self.engine.kill()
            self.engine = None
            self._pidfile(None)
        if self.guard is not None:
            problems = []
            if self.trip and time.monotonic() - self.trip[1] < self.guard.cfg.cooldown_s:
                left = self.guard.cfg.cooldown_s - (time.monotonic() - self.trip[1])
                problems.append(f"the memory watchdog unloaded the model {time.monotonic() - self.trip[1]:.0f}s ago "
                                f"({self.trip[0]}); not loading again for {left:.0f}s")
            problems += self.guard.preflight()
            if problems:
                raise EngineBlocked("not loading the model: " + "; ".join(problems))
            self.guard.arm()
        self.state = "loading"
        t0 = time.monotonic()
        eng = None
        try:
            eng = Engine(self.cfg, self.log)
            self._loading = eng
            self.log(f"engine starting: pid {eng.pid}: {' '.join(self.cfg.argv())}")
            eng.wait_ready(self.cfg.ready_timeout)
        except BaseException as e:
            self.state = "unloaded"
            if eng is not None and eng.tripped:
                raise MemoryTrip(f"the memory watchdog stopped the model while it loaded: {eng.tripped}") from e
            raise
        finally:
            self._loading = None
        self.engine, self.state = eng, "loaded"
        self.loads += 1
        self._pidfile(eng.pid)
        self.log(f"engine ready in {time.monotonic() - t0:.1f}s (ctx {eng.ctx}): {eng.info}")
        return eng

    def _pidfile(self, pid) -> None:
        if not self.cfg.pid_path:
            return
        try:
            if pid is None:
                os.unlink(self.cfg.pid_path)
            else:
                with open(self.cfg.pid_path, "w") as f:
                    f.write(f"{pid}\n")
        except OSError:
            pass

    def _unload(self, why: str) -> None:
        eng, self.engine = self.engine, None
        self.state = "unloaded"
        if eng is not None:
            eng.quit()
            self._pidfile(None)
            self.log(f"engine stopped ({why}); pid {eng.pid} exit code {eng.proc.poll()}")

    def _watch_loop(self) -> None:
        """Every poll_s while an engine is loading or loaded: unload it at once if memory is in trouble."""
        while not self._stop.wait(self.guard.poll_s):
            eng = self.engine or self._loading
            if eng is None or not eng.alive():
                continue
            try:
                reason = self.guard.check()
            except Exception as e:                    # a probe failing must not kill the watchdog
                self.log(f"memory watchdog: probe failed: {e}")
                continue
            if reason:
                self._trip(eng, reason)

    def _trip(self, eng: Engine, reason: str) -> None:
        self.trip = (reason, time.monotonic())
        eng.tripped = reason
        if self.engine is eng:
            self.engine = None
            self.state = "unloaded"
        self.log(f"MEMORY WATCHDOG: unloading the engine (pid {eng.pid}): {reason}")
        eng.kill()                                    # memory first: no QUIT, no waiting for the request
        self._pidfile(None)

    def _idle_loop(self) -> None:
        while not self._stop.wait(min(5.0, max(0.05, self.cfg.idle / 4)) if self.cfg.idle > 0 else 5.0):
            if self.cfg.idle <= 0 or self.engine is None:
                continue
            if time.monotonic() - self.last_used < self.cfg.idle or not self.lock.try_acquire():
                continue
            try:
                if self.engine is not None and time.monotonic() - self.last_used >= self.cfg.idle:
                    self._unload(f"idle {self.cfg.idle:.0f}s")
            finally:
                self.lock.release()

    def unload(self, timeout: float = 60.0) -> bool:
        """Stop the engine when the current request is done (POST /unload, tests)."""
        if not self._acquire_within(timeout):
            return False
        try:
            self._unload("asked")
        finally:
            self.lock.release()
        return True

    def _acquire_within(self, timeout: float) -> bool:
        end = time.monotonic() + timeout
        return self.lock.acquire(lambda: time.monotonic() > end, poll=0.1)

    def shutdown(self, timeout: float = 30.0) -> None:
        """Running requests see `closing` and cancel; then the engine is stopped. Never leaves a child behind."""
        self.closing = True
        self._stop.set()
        got = self._acquire_within(timeout)
        try:
            eng, self.engine = self.engine, None
            if eng is not None:
                if got:
                    eng.quit()
                else:
                    eng.kill()
                self._pidfile(None)
                self.log(f"engine stopped (shutdown); pid {eng.pid}")
        finally:
            if got:
                self.lock.release()
        self.state = "unloaded"
