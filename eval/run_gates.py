#!/usr/bin/env python3
"""One entry point for the acceptance gates: three checks of an edited model against references.

  gate 1  refusal rate: held-out harmful prompts (+ harmless controls), reasoning off and on, classified as
          hard refusal / soft refusal / reframed answer / complies (eval/lib/classify.py)
  gate 2  "not broken": KL and top-1 agreement against the original pack on WikiText-2 text, perplexity, and
          coherence on three long prompts
  gate 3  speed: prefill and decode tokens per second on real prompts and the engine's synthetic --bench

  uv run --no-project --with jinja2 --with tokenizers --with numpy python eval/run_gates.py --model-dir DIR \
      --pack PACK.strata [--gate 1|2|3|all]
  ... --pack Inkling-2-Mini-SQ-15.strata --engine PATCHED/strata-inkling --engine-args "--ablate DIRS.bin"
  ... --llama --llama-url http://127.0.0.1:8080     # gate 1 for a GGUF served by llama.cpp's llama-server

Run eval/prep_data.py once first (it downloads the prompt sets and WikiText-2). Writes <runs>/<name>/summary.md
(+ gate1/*.jsonl transcripts, gate2/, gate3/). Transcripts contain full model outputs: keep them private.

Each phase that loads a model re-invokes this script with --phase, optionally under an external guard command
(--guard; see lib/common.guarded for its calling convention and exit codes). With --guard-lock, every engine
start also asserts that the guard's lock is held by an ancestor process, so no code path starts an engine
unguarded. Without --guard, phases run directly, one after another.
"""
import argparse, json, os, re, shlex, subprocess, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# options that configure lib.common (exported as environment variables before it is imported, so that the
# --phase re-invocations see the same settings)
ENV_OPTS = {"--model-dir": "QE_MODEL_DIR", "--orig-pack": "QE_ORIG_PACK", "--strata-bin": "STRATA_BIN",
            "--kl-bin": "KL_BIN", "--data-dir": "QE_DATA_DIR", "--runs-dir": "QE_RUNS_DIR",
            "--refs-dir": "QE_REFS_DIR", "--llama-url": "LLAMA_URL", "--llama-ctl": "QE_LLAMA_CTL",
            "--llama-busy": "QE_LLAMA_BUSY", "--guard": "QE_GUARD", "--guard-lock": "QE_GUARD_LOCK",
            "--health-url": "QE_HEALTH_URL", "--watch-file": "QE_WATCH_FILE", "--watch-pattern": "QE_WATCH_PATTERN"}


def _export_env(argv):
    pre = argparse.ArgumentParser(add_help=False)
    for opt in ENV_OPTS:
        pre.add_argument(opt)
    known, _ = pre.parse_known_args(argv)
    for opt, var in ENV_OPTS.items():
        v = getattr(known, opt[2:].replace("-", "_"))
        if v is not None:
            if opt in ("--model-dir", "--orig-pack", "--data-dir", "--runs-dir", "--refs-dir", "--guard-lock",
                       "--watch-file", "--strata-bin", "--kl-bin") and os.path.exists(v):
                v = os.path.abspath(v)
            os.environ[var] = v


_export_env(sys.argv[1:])
from lib.common import (RUNS_DIR, ORIG_PACK, STRATA_BIN, StrataServer, LlamaClient, GuardError,  # noqa: E402
                        guarded, assert_guarded, llama_ctl, llama_busy, pids, log, health_state, LLAMA_CTL)
import gate1, gate2, gate3  # noqa: E402

TARGETS = {"p_4k": 1118, "p_28k": 800}     # earlier measurement, prefill tok/s (6,630 and 28,156 prompt tokens)


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:60]


def run_name(a):
    if a.name: return a.name
    if a.llama: return "llama-atomicchat-q4k"
    n = os.path.splitext(os.path.basename(a.pack))[0]
    if a.engine:
        n += "__" + slug(os.path.basename(os.path.dirname(os.path.abspath(a.engine))))
    if a.engine_args:
        n += "__" + slug(" ".join(os.path.basename(x) for x in shlex.split(a.engine_args)))
    return n


def fmt_kl(k):
    if not k: return "not run"
    s = (f"- positions: {k['positions']} ({k['windows']} WikiText-2 windows)\n"
         f"- **mean KL(orig || candidate): {k['kl_mean']:.5f}** (median {k['kl_median']:.5f}, p99 {k['kl_p99']:.4f}, max {k['kl_max']:.3f})\n"
         f"- **top-1 agreement: {100*k['top1_agreement']:.2f}%**\n"
         f"- perplexity: candidate {k['ppl_candidate']:.3f} vs original {k['ppl_reference']:.3f}\n"
         f"- reference tail mass outside top-{gate2.K}: {k['ref_tail_mass_mean']:.2e} (KL is a lumped-tail lower bound)\n"
         f"- engine `--kld-eval` on window 0 (exact, 511 positions): `{k.get('engine_kld_eval_w0')}`; harness window 0: KL {k.get('harness_w0', {}).get('kl', float('nan')):.5f}, top-1 {100*k.get('harness_w0', {}).get('top1', 0):.2f}%\n"
         f"- engine: `{k['engine']}` {' '.join(k.get('engine_args', []))}; {k['seconds']} s\n"
         f"- AtomicChat's own figure for the edit (BF16+adapter vs BF16): 0.0078 neutral text / 0.0076 code\n")
    return s


def fmt_coh(c):
    if not c: return "not run"
    out = ["| prompt | prompt tok | finish | distinct 4-grams | vs reference |", "|---|---|---|---|---|"]
    for n, r in c.items():
        ref = ("**reference recorded**" if "first_divergence" not in r else
               ("identical" if r.get("identical") else f"first divergence at token {r['first_divergence']} of {r['ref_len']}"))
        out.append(f"| {n} | {r['prompt_tokens']} | {r['finish']} | {r['distinct_4gram']:.2f} | {ref} |")
    return "\n".join(out) + "\n\nOutputs: `gate2/coherence_*.txt` (read them; divergence alone is expected for any edited model).\n"


def fmt_speed(real, bench):
    out = ["| measure | prompt tokens | prefill tok/s | decode tok/s | earlier measurement |", "|---|---|---|---|---|"]
    for n, r in (real or {}).items():
        reps = ", ".join(f"{x['prefill_tok_s']:.0f}" for x in r["reps"])
        out.append(f"| real `{n}` (serve, median of {len(r['reps'])}: {reps}) | {r['prompt_tokens']} | **{r['prefill_tok_s_median']:.0f}** | "
                   f"{r['decode_tok_s_median']:.1f} | {TARGETS.get(n, '~31 decode') if n in TARGETS else '–'} |")
    for n, b in (bench or {}).items():
        out.append(f"| synthetic `--bench --prompt-len {n} --gen 128 --draft off` | {n} | {b['prefill_tok_s']} | {b['decode_tok_s']} | |")
    return "\n".join(out) + "\n"


def write_summary(run_dir, a, parts, timings, health_log, status):
    s = [f"# Gate summary: `{os.path.basename(run_dir)}`\n", f"- status: **{status}**",
         f"- date: {time.strftime('%Y-%m-%d %H:%M')}",
         f"- target: {'llama.cpp server at ' + a.llama_url if a.llama else '`' + a.pack + '`'}",
         f"- engine: `{a.engine or STRATA_BIN}` {a.engine_args or ''}",
         f"- gate timings (wall, incl. lock waits): {', '.join(f'{k} {v/60:.1f} min' for k, v in timings.items())}",
         f"- health / watch-file checks: {'; '.join(health_log)}\n"]
    if "gate1" in parts:
        s += ["## Gate 1: refusal rate\n", parts["gate1"]]
        ref_dir = os.path.join(RUNS_DIR, "llama-atomicchat-q4k")
        if not a.llama and os.path.isdir(ref_dir):
            ref = gate1.score(ref_dir)
            ids = {st: {r["id"] for r in recs} for (st, m), recs in ref.items()}
            s += ["\n**Same prompt subset as the llama.cpp reference:**\n", gate1.table(gate1.score(run_dir, ids=ids)),
                  "\n**Reference (the llama.cpp run, the target):**\n", gate1.table(ref)]
        s.append("")
    if "kl" in parts or "coh" in parts:
        s += ["## Gate 2: not broken\n", "### Running-text KL / top-1 vs original SQ-15\n", parts.get("kl", "not run"),
              "### Coherence on the model folder's tools/prompts\n", parts.get("coh", "not run")]
    if "speed" in parts:
        s += ["## Gate 3: speed\n", parts["speed"]]
    open(os.path.join(run_dir, "summary.md"), "w").write("\n".join(s) + "\n")
    log(f"wrote {os.path.join(run_dir, 'summary.md')}")


PHASES = ("gate1", "gate1-llama", "gate2-kl", "gate2-coh", "gate3")


def phase_main(a, run_dir, engine, extra, modes, is_orig):
    """Runs as its own process (under the guard, if one is configured). Loads exactly one model (or uses the
    running llama-server)."""
    assert_guarded()
    ph = a.phase
    if ph == "gate1":
        os.makedirs(os.path.join(run_dir, "gate1"), exist_ok=True)
        # parallel=1: the stock engine's multi-slot GENERATE path returns token 0 ("!") as the first
        # token for concurrently admitted requests (seen 2026-11-22), so slots > 1 are never used.
        with StrataServer(a.pack, ctx=4096, parallel=1, engine=engine, extra=extra,
                          stderr_path=os.path.join(run_dir, "engine_gate1.stderr"),
                          trace=os.path.join(run_dir, "gate1", "protocol_trace.txt")) as S:
            gate1.run("strata", run_dir, modes=modes, limit=a.limit, server=S)
    elif ph == "gate1-llama":
        if not pids("llama-server"):
            llama_ctl("start")
        L = LlamaClient(); L.wait_ready()
        gate1.run("llama", run_dir, modes=modes, limit=a.limit, llama=L, limits={"harmful": a.llama_harmful})
    elif ph == "gate2-kl":
        g2 = os.path.join(run_dir, "gate2"); os.makedirs(g2, exist_ok=True)
        if is_orig and not os.path.exists(os.path.join(gate2.REF_DIR, "kl_ref.npz")):
            log("gate2: recording the original SQ-15 reference")
            gate2.build_ref(a.pack, g2)
        gate2.score_kl(a.pack, g2, g2, a.kl_windows, binary=engine if engine else None, extra=extra)
    elif ph == "gate2-coh":
        g2 = os.path.join(run_dir, "gate2"); os.makedirs(g2, exist_ok=True)
        with StrataServer(a.pack, ctx=32768, parallel=1, engine=engine, extra=extra,
                          stderr_path=os.path.join(run_dir, "engine_gate2.stderr")) as S:
            record = is_orig and not os.path.exists(os.path.join(gate2.REF_DIR, "coherence_p_short.json"))
            gate2.run_coherence(S, g2, record_ref=record)
    elif ph == "gate3":
        g3 = os.path.join(run_dir, "gate3"); os.makedirs(g3, exist_ok=True)
        with StrataServer(a.pack, ctx=32768, parallel=1, engine=engine, extra=extra,
                          stderr_path=os.path.join(run_dir, "engine_gate3.stderr")) as S:
            gate3.run_real(S, g3)
        gate3.run_bench(a.pack, g3, engine=engine, extra=extra)
    return 0


def main():
    ap = argparse.ArgumentParser(description="acceptance gates for an edited model (see the module docstring)")
    g = ap.add_argument_group("paths (also settable as environment variables, see lib/common.py)")
    g.add_argument("--model-dir", help="folder with tokenizer.json, chat_template.jinja, tools/prompts/ [QE_MODEL_DIR]")
    g.add_argument("--orig-pack", help="the original pack; gate-2 references come from it [QE_ORIG_PACK]")
    g.add_argument("--strata-bin", help="the stock strata-inkling [STRATA_BIN, default: on PATH]")
    g.add_argument("--kl-bin", help="strata-inkling with the 64-row fix, for gate-2 dumps of stock runs [KL_BIN]")
    g.add_argument("--data-dir", help="prep_data.py output [QE_DATA_DIR, default eval/data]")
    g.add_argument("--runs-dir", help="results and transcripts [QE_RUNS_DIR, default eval/runs]")
    g.add_argument("--refs-dir", help="gate-2 references [QE_REFS_DIR, default eval/refs]")
    g = ap.add_argument_group("llama.cpp reference")
    g.add_argument("--llama-url", help="llama-server URL [LLAMA_URL, default http://127.0.0.1:8080]")
    g.add_argument("--llama-ctl", help="optional command taking start/stop for that server [QE_LLAMA_CTL]")
    g.add_argument("--llama-busy", help="process pattern: never stop the server while it matches [QE_LLAMA_BUSY]")
    g = ap.add_argument_group("optional guard (off by default)")
    g.add_argument("--guard", help="command wrapping each model-loading phase (lib/common.guarded) [QE_GUARD]")
    g.add_argument("--guard-lock", help="the guard's lock file {\"pid\": N}; engine starts assert it [QE_GUARD_LOCK]")
    g.add_argument("--health-url", help="recorded in summary.md only [QE_HEALTH_URL]")
    g.add_argument("--watch-file", help="recorded in summary.md only: pattern count in this file [QE_WATCH_FILE]")
    g.add_argument("--watch-pattern", help="default 'insufficient memory' [QE_WATCH_PATTERN]")
    ap.add_argument("--pack", default=ORIG_PACK)
    ap.add_argument("--engine", help="strata-inkling binary (default: the stock binary; gate-2 KL dumps of a "
                                     "stock-binary run use KL_BIN)")
    ap.add_argument("--engine-args", default="", help="extra args passed to every engine invocation, e.g. \"--ablate D.bin\"")
    ap.add_argument("--gate", default="all", choices=["1", "2", "3", "all"])
    ap.add_argument("--llama", action="store_true", help="gate 1 for the llama.cpp reference (llama-server at --llama-url)")
    ap.add_argument("--llama-harmful", type=int, default=60, help="harmful prompts for the llama.cpp reference (time budget)")
    ap.add_argument("--name")
    ap.add_argument("--limit", type=int, help="gate 1: first N prompts of each set (smoke tests)")
    ap.add_argument("--kl-windows", type=int, help="gate 2: first N of the 40 WikiText windows")
    ap.add_argument("--modes", default="off,on")
    ap.add_argument("--need-gb", type=float, default=18)
    ap.add_argument("--no-restore", action="store_true", help="(the default) don't start the llama-server at the end")
    ap.add_argument("--restore", action="store_true",
                    help="start the llama-server (--llama-ctl start) at the end if it isn't running")
    ap.add_argument("--phase", choices=PHASES, help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.llama_url = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
    a.pack = os.path.abspath(a.pack)
    extra = shlex.split(a.engine_args)
    engine = os.path.abspath(a.engine) if a.engine else None
    run_dir = os.path.join(RUNS_DIR, run_name(a)); os.makedirs(run_dir, exist_ok=True)
    modes = tuple(m for m in a.modes.split(",") if m)
    is_orig = (a.pack == os.path.abspath(ORIG_PACK) and not engine and not extra)
    if a.phase:
        return phase_main(a, run_dir, engine, extra, modes, is_orig)

    gates = {"1", "2", "3"} if a.gate == "all" else {a.gate}
    if a.llama: gates = {"1"}
    passthru = [a for a in sys.argv[1:]]
    me = [sys.executable, os.path.abspath(__file__), *passthru, "--name", os.path.basename(run_dir)]
    parts, timings, health_log = {}, {}, ["start %s/%s" % health_state()]
    status = "complete"
    stopped_llama = [False]
    log(f"run dir {run_dir}; gates {sorted(gates)}")

    def free_llama_for_exclusive():
        """Exclusive speed runs need the llama-server down; stop it only if no foreign client is using it."""
        if pids("llama-server") and not llama_busy():
            llama_ctl("stop"); stopped_llama[0] = True
        elif pids("llama-server"):
            log("speed gate waits: llama-server is in use by another client")

    def phase(name, need, mode="strata", exclusive=False, hook=None):
        t = time.time()
        rc = guarded(f"{os.path.basename(run_dir)[:24]}-{name}", need, me + ["--phase", name], mode=mode,
                     exclusive=exclusive, on_refused=hook, refused_retries=60 if exclusive else 30)
        timings[name] = time.time() - t
        health_log.append(f"after {name}: %s/%s" % health_state())
        if rc != 0:
            raise RuntimeError(f"phase {name} exited {rc}")

    try:
        if a.llama:
            phase("gate1-llama", 2 if pids("llama-server") else 20, mode="llama")
        else:
            if "2" in gates:
                phase("gate2-kl", a.need_gb)
                phase("gate2-coh", a.need_gb)
            if "3" in gates:
                phase("gate3", a.need_gb, exclusive=True, hook=free_llama_for_exclusive)
            if "1" in gates:
                phase("gate1", a.need_gb)
    except GuardError as e:
        status = f"ABORTED by the guard: {e}"; log(status)
    except Exception as e:
        status = f"FAILED: {e!r}"; log(status); traceback.print_exc()

    # Default: never start the llama-server here (a loaded large-context llama-server may leave no room for a
    # Strata pack). Only --restore starts it, through --llama-ctl.
    if a.restore and not a.no_restore and LLAMA_CTL and not pids("llama-server") and not status.startswith("ABORTED"):
        try:
            guarded("llama-start", 20, shlex.split(LLAMA_CTL) + ["start"], mode="llama")
        except GuardError as e:
            status += f"; llama-server restart refused: {e}"

    def jload(*p):
        q = os.path.join(run_dir, *p)
        return json.load(open(q)) if os.path.exists(q) else None
    # summary.md always reflects every gate with results in run_dir (separate invocations accumulate)
    tpath = os.path.join(run_dir, "timings.json")
    allt = jload("timings.json") or {}
    allt.update({k: round(v) for k, v in timings.items()}); json.dump(allt, open(tpath, "w"), indent=1)
    timings = allt
    if os.path.isdir(os.path.join(run_dir, "gate1")) and any(f.endswith(".jsonl") for f in os.listdir(os.path.join(run_dir, "gate1"))):
        try:
            rows = gate1.score(run_dir, a.limit); parts["gate1"] = gate1.table(rows) + "\n" + gate1.examples(rows)
        except Exception as e: parts["gate1"] = f"scoring failed: {e!r}"
    if jload("gate2", "kl.json"): parts["kl"] = fmt_kl(jload("gate2", "kl.json"))
    if jload("gate2", "coherence.json"): parts["coh"] = fmt_coh(jload("gate2", "coherence.json"))
    if jload("gate3", "speed_real.json") or jload("gate3", "speed_bench.json"):
        parts["speed"] = fmt_speed(jload("gate3", "speed_real.json"), jload("gate3", "speed_bench.json"))
    write_summary(run_dir, a, parts, timings, health_log, status)
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
