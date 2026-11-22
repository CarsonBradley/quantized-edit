"""A stand-in for `strata-inkling --serve` that speaks its stdin protocol without a GPU (tests only).

Mirrors ink_serve.h where the shim depends on it: INFO/READY on start; GENERATE <rid> <max_new> [k=v..] <ids>;
all-or-nothing prefix reuse of the slot's fed tokens (RESUME n, capped at prompt_len - 1); PP lines per prefill
chunk; T <rid> <id> per token; the last sampled token is not fed back; CANCEL <rid> between steps;
DONE <rid> gen plen prompt_ms dec_ms reason; QUIT / stdin EOF exit once nothing is running.

Behaviour comes from a JSON file named by FAKE_SCRIPT:
    {"replies": [[ids...], ...],   each GENERATE takes the next reply; the last one repeats
     "token_delay": 0.0, "chunk_delay": 0.0, "load_delay": 0.0, "fail_load": false,
     "crash_after": null,          exit(3) after this many tokens of a reply (simulated GPU fault)
     "gate": null}                 a file path: while it exists, the engine waits before every prefill chunk and
                                   every token (CANCEL still works), so tests can hold a request mid-flight
Every command is appended to FAKE_LOG as one JSON object per line.
"""
import json
import os
import queue
import sys
import threading
import time

STOP_IDS = (200006, 199999)


def main():
    args = sys.argv[1:]
    opt = lambda k, d: args[args.index(k) + 1] if k in args else d  # noqa: E731
    max_ctx = int(opt("--max-context", 32768))
    chunk = int(opt("--chunk", 512))
    script = json.load(open(os.environ["FAKE_SCRIPT"])) if os.environ.get("FAKE_SCRIPT") else {}
    logf = open(os.environ["FAKE_LOG"], "a") if os.environ.get("FAKE_LOG") else None

    def log(**kw):
        if logf:
            logf.write(json.dumps(kw) + "\n")
            logf.flush()

    def out(line):
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    log(cmd="START", argv=args, pid=os.getpid())
    time.sleep(script.get("load_delay", 0))
    if script.get("fail_load"):
        out("ERR open: fake load failure")
        return 1
    out(f"INFO engine=strata-inkling-0.1.65 arch=inkling2_loop backend=fake model={os.path.basename(opt('--native', '?'))} "
        f"layers=24 visits=48 vocab=200058 chunk={chunk} batch_slots=1 slot_read=1 kv_dtype=fp16 drafter=none images=0")
    out(f"READY {max_ctx} stop hold holdb")

    lines = queue.Queue()

    def reader():
        for line in sys.stdin:
            lines.put(line.rstrip("\r\n"))
        lines.put(None)
    threading.Thread(target=reader, daemon=True).start()

    replies = script.get("replies") or [[200001, 200004, 13225, 200010, 200006]]
    n_req = 0
    cached = []
    quit_after = False
    cancels = set()

    def poll():
        nonlocal quit_after
        while True:
            try:
                line = lines.get_nowait()
            except queue.Empty:
                return
            if line is None or line == "QUIT":
                quit_after = True
            elif line.startswith("CANCEL "):
                cancels.add(line[7:])
                log(cmd="CANCEL", rid=line[7:])

    def held(rid):
        """Wait while the gate file exists; True if the request was cancelled meanwhile."""
        gate = script.get("gate")
        while gate and os.path.exists(gate) and rid not in cancels:
            time.sleep(0.02)
            poll()
        return rid in cancels

    while not quit_after:
        line = lines.get()
        if line is None or line == "QUIT":
            log(cmd="QUIT")
            break
        if line.startswith("CANCEL "):
            log(cmd="CANCEL", rid=line[7:], idle=True)
            continue
        if not line.startswith("GENERATE "):
            out("ERR unknown command")
            continue
        w = line.split(" ")
        rid, max_new, keys, ids = w[1], int(w[2]), dict(k.split("=", 1) for k in w[3:-1]), list(map(int, w[-1].split(",")))
        if len(ids) >= max_ctx:
            out(f"ERR prompt length out of range (1..{max_ctx - 1})")
            continue
        lim = min(len(cached), len(ids) - 1)
        reuse = 0
        while reuse < lim and cached[reuse] == ids[reuse]:
            reuse += 1
        if reuse != len(cached):
            reuse = 0
        log(cmd="GENERATE", rid=rid, max_new=max_new, keys=keys, ids=ids, reuse=reuse, cached=len(cached))
        if reuse:
            out(f"RESUME {reuse}")
        else:
            cached = []
        reply = replies[min(n_req, len(replies) - 1)]
        n_req += 1
        t0 = time.monotonic()
        pos, reason = reuse, None
        while pos < len(ids):
            if held(rid):
                reason = "cancel"
                break
            take = min(chunk, len(ids) - pos)
            time.sleep(script.get("chunk_delay", 0))
            cached += ids[pos:pos + take]
            pos += take
            ms = (time.monotonic() - t0) * 1000
            out(f"PP {pos} {len(ids)} {ms:.1f} {take / max(ms, 1e-3) * 1000:.1f}")
            poll()
            if rid in cancels:
                reason = "cancel"
                break
        prompt_ms = (time.monotonic() - t0) * 1000
        t1 = time.monotonic()
        gen = 0
        if reason is None and max_new == 0:
            reason = "length"
        crash = script.get("crash_after")
        while reason is None:
            if gen > 0:
                cached.append(reply[gen - 1])            # the previous token is fed now
            if gen >= len(reply):
                reason = "length"
                break
            if held(rid):
                reason = "cancel"
                break
            tok = reply[gen]
            out(f"T {rid} {tok}")
            gen += 1
            if crash is not None and gen >= crash:
                log(cmd="CRASH")
                os._exit(3)
            if tok in STOP_IDS:
                reason = "stop"
            elif gen >= max_new or len(cached) + 1 >= max_ctx:
                reason = "length"
            else:
                time.sleep(script.get("token_delay", 0))
                poll()
                if rid in cancels:
                    reason = "cancel"
        out(f"DONE {rid} {gen} {len(ids)} {prompt_ms:.1f} {(time.monotonic() - t1) * 1000:.1f} {reason}")
        log(cmd="DONE", rid=rid, gen=gen, reason=reason, cached=len(cached))
    log(cmd="EXIT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
