"""Gate 3: speed. Real chat prompts (the model folder's tools/prompts/) through --serve, GEN, temperature 0,
--draft off, plus the engine's synthetic --bench at the same prompt lengths.

prefill tok/s = prompt_tokens / prompt_ms  (DONE line);  decode tok/s = generated / dec_ms.
Each real prompt is run REPS times in one engine session (an identical repeated prompt does not hit
the prefix cache: the slot's cache holds prompt+generated tokens, which is longer than the new prompt,
so the engine resets; verified by the per-rep prefill times). Median is reported.
"""
import json, os, re, statistics, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.common import ROOT, STRATA_BIN, Renderer, assert_guarded, log, check_ablate_marker

PROMPTS = {"p_4k": "tools/prompts/p_4k.txt", "p_28k": "tools/prompts/p_28k.txt", "p_short": "tools/prompts/p_short.txt"}
REPS = 2
GEN_TOKENS = 256
BENCH_LENS = (6630, 28156)


def run_real(server, out_dir, prior=None):
    """server: started StrataServer, ctx >= 32768, parallel 1. prior: coherence results (count as rep 1)."""
    rnd = Renderer()
    res = {}
    for name, path in PROMPTS.items():
        ids = rnd.ids(open(os.path.join(ROOT, path)).read(), reasoning=True)
        reps = []
        if prior and name in prior:
            p = prior[name]
            reps.append({"plen": p["prompt_tokens"], "prefill_tok_s": p["prefill_tok_s"], "decode_tok_s": p["decode_tok_s"],
                         "gen": len(p["tokens"]), "src": "coherence run (512 new)"})
        while len(reps) < REPS:
            toks, gen, plen, pms, dms, why = server.gen(ids, GEN_TOKENS)
            reps.append({"plen": plen, "prefill_ms": pms, "prefill_tok_s": plen / (pms / 1000),
                         "gen": gen, "dec_ms": dms, "decode_tok_s": gen / (dms / 1000) if dms else None, "finish": why})
            log(f"speed {name}: prefill {plen} tok {plen/(pms/1000):.0f} tok/s, decode {gen} tok "
                f"{(gen/(dms/1000) if dms else 0):.1f} tok/s")
        res[name] = {"prompt_tokens": reps[0]["plen"], "reps": reps,
                     "prefill_tok_s_median": statistics.median(r["prefill_tok_s"] for r in reps),
                     "decode_tok_s_median": statistics.median(r["decode_tok_s"] for r in reps if r["decode_tok_s"])}
    json.dump(res, open(os.path.join(out_dir, "speed_real.json"), "w"), indent=1)
    return res


def run_bench(pack, out_dir, engine=None, extra=()):
    assert_guarded()
    res = {}
    for n in BENCH_LENS:
        cmd = [engine or STRATA_BIN, "--bench", "--native", pack, "--prompt-len", str(n), "--gen", "128",
               "--max-context", "32768", "--draft", "off", *extra]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        check_ablate_marker(extra, r.stderr, f"bench {n}")
        line = next((l for l in r.stdout.splitlines() if l.startswith("[bench]")), r.stdout[-300:] + r.stderr[-300:])
        m = re.search(r"prefill=([\d.]+) ppts.*decode=([\d.]+) tps", line)
        res[str(n)] = {"line": line, "prefill_tok_s": float(m.group(1)) if m else None, "decode_tok_s": float(m.group(2)) if m else None}
        log(f"bench {n}: {line}")
    json.dump(res, open(os.path.join(out_dir, "speed_bench.json"), "w"), indent=1)
    return res
