"""Shared pieces for the acceptance-gate harness (eval/run_gates.py).

- chat rendering (the model folder's chat_template.jinja + tokenizer.json), identical token ids for both engines
- the optional guard protocol (an external command that wraps every model-loading phase, e.g. a GPU/memory lock)
- a Strata `--serve` client (GENERATE/CANCEL over stdin, optional parallel slots)
- a llama.cpp client (raw token ids to /completion, streamed, early cancel)

Configuration comes from environment variables, which run_gates.py sets from its command-line options (so the
phases it re-invokes see the same settings):
  QE_MODEL_DIR    folder with tokenizer.json, chat_template.jinja and tools/prompts/ (the pack's Hugging Face folder)
  QE_ORIG_PACK    the original pack (default QE_MODEL_DIR/Inkling-2-Mini-SQ-15.strata); gate-2 references are
                  recorded from it
  STRATA_BIN      the stock strata-inkling binary (default: strata-inkling on PATH)
  KL_BIN          a strata-inkling with kldeval-64row-fix.patch, for gate-2 dumps (default: STRATA_BIN)
  QE_DATA_DIR     where eval/prep_data.py wrote the prompt sets and WikiText windows (default eval/data)
  QE_RUNS_DIR     per-run output, transcripts included (default eval/runs); QE_REFS_DIR references (default eval/refs)
  LLAMA_URL       llama-server for the llama.cpp reference (default http://127.0.0.1:8080)
  QE_LLAMA_CTL    optional command taking `start` / `stop` that manages that llama-server
  QE_LLAMA_BUSY   optional process-name pattern; while it matches, the llama-server is never stopped
  QE_GUARD        optional guard command (see `guarded`)
  QE_GUARD_LOCK   optional JSON file {"pid": N} the guard writes while it holds its lock; when set, every engine
                  start checks that the pid is an ancestor of this process (so no engine starts unguarded)
  QE_HEALTH_URL, QE_WATCH_FILE, QE_WATCH_PATTERN
                  optional, recorded in summary.md only: a health URL's status and the number of WATCH_PATTERN
                  occurrences in WATCH_FILE (e.g. another service's log and its out-of-memory message)
"""
import json, os, re, shlex, subprocess, sys, threading, time, urllib.request, urllib.error, urllib.parse, queue
from jinja2 import Environment
from jinja2.ext import loopcontrols
from tokenizers import Tokenizer


def _env(name, default=None):
    v = os.environ.get(name)
    return default if v in (None, "") else v


EVAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # this repo's eval/
ROOT = _env("QE_MODEL_DIR", os.getcwd())                             # the model folder
STRATA_BIN = _env("STRATA_BIN", "strata-inkling")
ORIG_PACK = _env("QE_ORIG_PACK", os.path.join(ROOT, "Inkling-2-Mini-SQ-15.strata"))
DATA_DIR = _env("QE_DATA_DIR", os.path.join(EVAL, "data"))
RUNS_DIR = _env("QE_RUNS_DIR", os.path.join(EVAL, "runs"))
REFS_DIR = _env("QE_REFS_DIR", os.path.join(EVAL, "refs"))
LLAMA_URL = _env("LLAMA_URL", "http://127.0.0.1:8080")
LLAMA_CTL = _env("QE_LLAMA_CTL")
LLAMA_BUSY = _env("QE_LLAMA_BUSY")
GUARD = _env("QE_GUARD")
GUARD_LOCK = _env("QE_GUARD_LOCK")
HEALTH_URL = _env("QE_HEALTH_URL")
WATCH_FILE = _env("QE_WATCH_FILE")
WATCH_PATTERN = _env("QE_WATCH_PATTERN", "insufficient memory")

# special token ids (tokenizer_config.json)
ENDOFTEXT, MSG_USER, MSG_MODEL, MSG_SYSTEM = 199999, 200000, 200001, 200002
CONTENT_TEXT, CONTENT_THINKING, END_MESSAGE, END_SAMPLING = 200004, 200008, 200010, 200006
STOP_IDS = {END_SAMPLING, ENDOFTEXT}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- rendering
class Renderer:
    def __init__(self, mdir=None):
        mdir = mdir or ROOT
        self.tok = Tokenizer.from_file(f"{mdir}/tokenizer.json")
        src = open(f"{mdir}/chat_template.jinja").read()
        src = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", src)
        env = Environment(extensions=[loopcontrols], trim_blocks=True, lstrip_blocks=True)
        def _raise(m): raise ValueError(m)
        env.globals["raise_exception"] = _raise
        self.tmpl = env.from_string(src)

    def text(self, user, reasoning=True):
        kw = {} if reasoning else {"reasoning_effort": "none"}   # "none" -> "Thinking effort level: 0"
        return self.tmpl.render(messages=[{"role": "user", "content": user}],
                                add_generation_prompt=True, bos_token="", eos_token="", **kw)

    def ids(self, user, reasoning=True):
        return self.tok.encode(self.text(user, reasoning), add_special_tokens=False).ids

    def decode(self, ids, special=True):
        return self.tok.decode(ids, skip_special_tokens=not special)


# ---------------------------------------------------------------- guard protocol
class GuardError(RuntimeError):
    pass


def health_state():
    """(health status code, or '-' when no QE_HEALTH_URL; WATCH_PATTERN count in QE_WATCH_FILE, or '-')."""
    code = "-"
    if HEALTH_URL:
        try:
            code = urllib.request.urlopen(HEALTH_URL, timeout=5).status
        except urllib.error.HTTPError as e:
            code = e.code
        except Exception:
            code = 0
    n = "-"
    if WATCH_FILE:
        try:
            n = open(WATCH_FILE, "rb").read().lower().count(WATCH_PATTERN.lower().encode())
        except FileNotFoundError:
            n = -1
    return code, n


def check_guard(stage):
    """Progress marker only; an external guard (if any) does the actual watching."""
    assert_guarded()


def pids(pattern):
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout
    return [int(p) for p in out.split() if int(p) != os.getpid()]


def llama_ctl(cmd):
    """Start or stop the llama-server through QE_LLAMA_CTL (no-op without it)."""
    if not LLAMA_CTL:
        log(f"llama-server {cmd}: no QE_LLAMA_CTL / --llama-ctl, skipped")
        return 0
    r = subprocess.run(shlex.split(LLAMA_CTL) + [cmd], capture_output=True, text=True, timeout=300)
    log(f"llama ctl {cmd}: {(r.stdout + r.stderr).strip()}")
    return r.returncode


def llama_clients():
    """Established client connections to the llama-server port other than llama-server itself."""
    port = urllib.parse.urlparse(LLAMA_URL).port or 80
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:ESTABLISHED"], capture_output=True, text=True).stdout
    return [l for l in out.splitlines()[1:] if not l.startswith("llama-ser")]


def llama_busy():
    """True while a foreign client uses the llama-server (connections, or a QE_LLAMA_BUSY process)."""
    return bool(llama_clients()) or bool(LLAMA_BUSY and pids(LLAMA_BUSY))


def _ancestors():
    out = subprocess.run(["/bin/ps", "-axo", "pid=,ppid="], capture_output=True, text=True).stdout
    parent = {}
    for line in out.splitlines():
        try:
            a, b = map(int, line.split()); parent[a] = b
        except ValueError:
            pass
    chain, p = set(), os.getpid()
    while p in parent and p not in chain and p > 1:
        chain.add(p); p = parent[p]
    return chain


def assert_guarded():
    """Every code path that loads a model calls this. With QE_GUARD_LOCK set, the guard's lock must be held by an
    ancestor process; without it there is nothing to check."""
    if not GUARD_LOCK:
        return
    try:
        owner = json.load(open(GUARD_LOCK))["pid"]
    except Exception:
        owner = None
    if owner is None or owner not in _ancestors():
        raise GuardError(f"not running under the guard (the lock {GUARD_LOCK} is not held by this process tree)")


def guarded(name, need_gb, argv, mode="strata", exclusive=False, max_wait_s=7200, refused_retries=30, on_refused=None):
    """Run argv, under QE_GUARD if set. The guard is called as
        QE_GUARD --name NAME --need-gb GB --mode strata|llama [--exclusive] -- ARGV...
    and may exit 75 (busy: wait 60 s, retry), 76 (refused: optional hook, wait 60 s, retry up to
    refused_retries) or 99 (killed: GuardError); any other code is the job's own. Returns the exit code."""
    if not GUARD:
        log(f"run {name} (no guard)")
        return subprocess.run(argv).returncode
    cmd = shlex.split(GUARD) + ["--name", name, "--need-gb", str(need_gb), "--mode", mode] + \
        (["--exclusive"] if exclusive else []) + ["--", *argv]
    t0, refused = time.time(), 0
    while True:
        log(f"guard {name} (need {need_gb} GB, mode {mode}{', exclusive' if exclusive else ''})")
        rc = subprocess.run(cmd).returncode
        if rc == 75:
            if time.time() - t0 > max_wait_s:
                raise GuardError(f"{name}: guard still busy after {max_wait_s}s")
            log(f"{name}: guard busy; retry in 60 s"); time.sleep(60); continue
        if rc == 76:
            refused += 1
            if refused > refused_retries:
                raise GuardError(f"{name}: guard refused {refused} times")
            if on_refused: on_refused()
            log(f"{name}: guard refused ({refused}/{refused_retries}); retry in 60 s"); time.sleep(60); continue
        if rc == 99:
            raise GuardError(f"{name}: killed by the guard")
        return rc


# ---------------------------------------------------------------- ablation-flag assertion
def check_ablate_marker(extra, text, where):
    """With --ablate in the engine args, the engine must print '[ink_metal] ablate ON: ...' (the patched build);
    otherwise a silently ignored flag would be scored as the candidate."""
    if "--ablate" in list(extra) and "ablate ON" not in (text or ""):
        raise RuntimeError(f"{where}: --ablate given but the engine did not print 'ablate ON' -- refusing to score")
    if "--ablate" in list(extra):
        line = next((l for l in text.splitlines() if "ablate ON" in l), "")
        log(f"{where}: {line.strip()}")


def read_from(path, offset):
    try:
        with open(path, errors="replace") as f:
            f.seek(offset); return f.read()
    except OSError:
        return ""


# ---------------------------------------------------------------- Strata --serve client
class StrataServer:
    """strata-inkling --serve, driven with GENERATE/CANCEL (works for 1..N slots)."""

    def __init__(self, pack, ctx=8192, parallel=1, draft="off", extra=(), stderr_path=None, engine=None, trace=None):
        self.cmd = [engine or STRATA_BIN, "--serve", "--native", pack, "--max-context", str(ctx),
                    "--parallel", str(parallel), "--draft", draft, "--kv", "fp16", *extra]
        self.parallel = parallel
        self.extra = list(extra)
        self.stderr_path = stderr_path or os.devnull
        self.p = None
        self.info = ""
        self.lines = queue.Queue()
        self.trace = open(trace, "a") if trace else None

    def start(self, timeout=300):
        assert_guarded()
        self._err = open(self.stderr_path, "a")
        self._err_off = self._err.tell()
        log("start:", " ".join(self.cmd))
        self.p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self._err, text=True, bufsize=1)
        threading.Thread(target=self._reader, daemon=True).start()
        t0 = time.time()
        while True:
            line = self._next(timeout - (time.time() - t0))
            if line.startswith("INFO"):
                self.info = line
            elif line.startswith("READY"):
                break
            elif line.startswith("ERR"):
                raise RuntimeError(line)
        log(f"strata ready in {time.time()-t0:.1f}s: {self.info}")
        if self.stderr_path != os.devnull:
            time.sleep(0.5)
            check_ablate_marker(self.extra, read_from(self.stderr_path, self._err_off), "serve")
        elif "--ablate" in self.extra:
            raise RuntimeError("--ablate runs need a stderr_path so the 'ablate ON' line can be checked")
        return self

    def _reader(self):
        for line in self.p.stdout:
            if self.trace: self.trace.write(f"{time.time():.3f} < {line}"); self.trace.flush()
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def _next(self, timeout=600):
        try:
            line = self.lines.get(timeout=max(1, timeout))
        except queue.Empty:
            raise RuntimeError("strata: timeout waiting for output")
        if line is None:
            raise RuntimeError(f"strata exited (rc={self.p.poll()})")
        return line

    def send(self, s):
        if self.trace: self.trace.write(f"{time.time():.3f} > {s[:200]}\n"); self.trace.flush()
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def run_jobs(self, jobs, on_progress=None):
        """jobs: iterable of objects with .next_request() -> (ids, max_new) or None,
        .on_token(tok) -> bool(cancel), .on_done(gen, plen, prompt_ms, dec_ms, reason).
        A job may issue several requests (phase 2). Runs up to self.parallel requests at once."""
        pending = list(jobs)
        active = {}
        seq = 0
        while pending or active:
            while pending and len(active) < self.parallel:
                job = pending.pop(0)
                req = job.next_request()
                if req is None:
                    if on_progress: on_progress(job)
                    continue
                ids, max_new = req
                rid = f"r{seq}"; seq += 1
                active[rid] = job
                job._cancelled = False
                self.send(f"GENERATE {rid} {max_new} temperature=0 {','.join(map(str, ids))}")
            if not active:
                continue
            line = self._next(600)
            if line.startswith("T "):
                _, rid, tok = line.split()
                job = active.get(rid)
                if job is not None and not job._cancelled and job.on_token(int(tok)):
                    job._cancelled = True
                    self.send(f"CANCEL {rid}")
            elif line.startswith("DONE "):
                f = line.split()
                rid = f[1]
                job = active.pop(rid, None)
                if job is not None:
                    job.on_done(int(f[2]), int(f[3]), float(f[4]), float(f[5]), f[6])
                    pending.insert(0, job)          # it may have a follow-up request
            elif line.startswith("ERR"):
                raise RuntimeError(f"strata: {line}")
            # RESUME / other lines: ignore

    def gen(self, ids, max_new):
        """Single blocking GEN (solo protocol). Returns (tokens, gen, plen, prompt_ms, dec_ms, reason)."""
        self.send(f"GEN {max_new} temperature=0 {','.join(map(str, ids))}")
        out = []
        while True:
            line = self._next(3600)
            if line.startswith("T "):
                out.append(int(line.split()[1]))
            elif line.startswith("DONE"):
                f = line.split()
                return out, int(f[1]), int(f[2]), float(f[3]), float(f[4]), f[5]
            elif line.startswith("ERR"):
                raise RuntimeError(f"strata: {line}")

    def close(self):
        if self.p and self.p.poll() is None:
            try:
                self.send("QUIT")
                self.p.wait(timeout=60)
            except Exception:
                self.p.kill()
                self.p.wait()
        if self.p:
            log(f"strata stopped (rc={self.p.returncode})")
        try: self._err.close()
        except Exception: pass

    def __enter__(self): return self.start()
    def __exit__(self, *a): self.close()


# ---------------------------------------------------------------- llama.cpp client
class LlamaClient:
    """llama-server /completion with raw token ids, streamed so we can stop early."""

    def __init__(self, url=LLAMA_URL):
        self.url = url

    def wait_ready(self, timeout=600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if urllib.request.urlopen(self.url + "/health", timeout=5).status == 200:
                    return True
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError("llama-server not healthy")

    def tokenize(self, text):
        body = json.dumps({"content": text, "add_special": False, "parse_special": True}).encode()
        r = urllib.request.urlopen(urllib.request.Request(self.url + "/tokenize", body,
                                   {"Content-Type": "application/json"}), timeout=120)
        return [t if isinstance(t, int) else t["id"] for t in json.load(r)["tokens"]]

    def run_job(self, job):
        assert_guarded()
        while True:
            req = job.next_request()
            if req is None:
                return
            ids, max_new = req
            body = {"prompt": ids, "n_predict": max_new, "temperature": 0, "top_k": 1,
                    "stream": True, "return_tokens": True, "cache_prompt": True}
            r = urllib.request.urlopen(urllib.request.Request(self.url + "/completion", json.dumps(body).encode(),
                                       {"Content-Type": "application/json"}), timeout=3600)
            gen, reason, timings = 0, "length", {}
            cancelled = False
            try:
                for raw in r:
                    raw = raw.decode("utf-8", "replace").strip()
                    if not raw.startswith("data: "):
                        continue
                    d = json.loads(raw[6:])
                    for t in d.get("tokens", []) or []:
                        gen += 1
                        if job.on_token(int(t)):
                            cancelled = True
                            break
                    if cancelled:
                        reason = "cancel"
                        break
                    if d.get("stop"):
                        timings = d.get("timings", {}) or {}
                        reason = "stop" if d.get("stop_type") in ("eos", "word") else "length"
                        break
            finally:
                r.close()                      # closing the stream cancels the task server-side
            job.on_done(gen, len(ids), float(timings.get("prompt_ms", 0)), float(timings.get("predicted_ms", 0)), reason)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))
