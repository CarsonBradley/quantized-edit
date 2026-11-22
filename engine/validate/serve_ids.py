"""serve_ids.py - drive `strata-inkling --serve` with one chat prompt, greedy, and save the token ids.

usage: uv run --no-project --with jinja2 --with tokenizers python -I serve_ids.py ENGINE PACK MODEL_DIR PROMPT_FILE MAX_NEW OUT.json
                                                                  [--effort none|low|...|0.9] [engine args ...]
  MODEL_DIR holds tokenizer.json + chat_template.jinja (the pack's Hugging Face folder).
  PROMPT_FILE: a text file with the user message, or a .jsonl/.json list (see --many below).
  Extra engine args are passed through, e.g. --draft off --max-context 8192 --ablate DIRS.bin --ablate-scale 0
  --many: PROMPT_FILE is a JSON list of strings; all prompts run in ONE engine process (one model load),
          one after another, and OUT.json holds a list of results.
Writes {"ids": [...], "done": "...", "prompt_tokens": N, "text": "...", "stderr_tail": [...], "argv": [...]}.
Exit code 0 only if the engine answered every prompt with DONE.
"""
import json
import re
import subprocess
import sys
import time

from jinja2 import Environment
from jinja2.ext import loopcontrols
from tokenizers import Tokenizer


def main():
    engine, pack, mdir, pfile, max_new, out_json = sys.argv[1:7]
    rest = sys.argv[7:]
    effort, many = None, False
    while rest[:1] in (["--effort"], ["--many"]):
        if rest[0] == "--many":
            many = True
            rest = rest[1:]
        else:
            effort = rest[1]
            rest = rest[2:]
            try:
                effort = float(effort)
            except ValueError:
                pass
    tok = Tokenizer.from_file(f"{mdir}/tokenizer.json")
    tmpl = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", open(f"{mdir}/chat_template.jinja").read())

    def _raise(m):
        raise ValueError(m)

    env = Environment(extensions=[loopcontrols], trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = _raise
    kw = {} if effort is None else {"reasoning_effort": effort}
    prompts = json.load(open(pfile)) if many else [open(pfile).read()]
    all_ids = []
    for pr in prompts:
        text = env.from_string(tmpl).render(messages=[{"role": "user", "content": pr}],
                                            add_generation_prompt=True, bos_token="", eos_token="", **kw)
        all_ids.append(tok.encode(text, add_special_tokens=False).ids)
    argv = [engine, "--serve", "--native", pack, *rest]
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    t0 = time.time()
    ready = False
    for line in p.stdout:
        line = line.strip()
        if line.startswith("ERR"):
            print(line)
            break
        if line.startswith("READY"):
            ready = True
            break
    if not ready:
        p.kill()
        print("engine did not become READY:\n" + p.stderr.read()[-2000:])
        sys.exit(1)
    load_s = time.time() - t0
    results, ok = [], True
    for ids in all_ids:
        p.stdin.write(f"GEN {max_new} temperature=0 {','.join(map(str, ids))}\n")
        p.stdin.flush()
        out, done = [], None
        for line in p.stdout:
            line = line.strip()
            if line.startswith("T "):
                out.append(int(line.split()[1]))
            elif line.startswith("DONE") or line.startswith("ERR"):
                done = line
                break
        ok = ok and bool(done) and done.startswith("DONE")
        results.append({"prompt_tokens": len(ids), "done": done, "ids": out,
                        "text": tok.decode(out, skip_special_tokens=False)})
        f = (done or "").split()
        if len(f) >= 6 and f[0] == "DONE":
            g, pl, pms, dms = int(f[1]), int(f[2]), float(f[3]), float(f[4])
            print(f"prompt {pl} tok, prefill {pl / max(pms, 1e-9) * 1e3:.0f} tok/s, "
                  f"gen {g} tok at {g / max(dms, 1e-9) * 1e3:.1f} tok/s, stop={f[5]}")
        else:
            print(f"no DONE: {done}")
    try:
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        p.wait(timeout=60)
    except Exception:
        p.kill()
    err = p.stderr.read().strip().splitlines()
    meta = {"argv": argv, "load_s": round(load_s, 1), "stderr_tail": err[-6:]}
    if many:
        json.dump({**meta, "results": results}, open(out_json, "w"), indent=1)
    else:
        json.dump({**meta, **results[0]}, open(out_json, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
