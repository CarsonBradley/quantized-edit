"""smoke_refusal.py - refusal yes/no per prompt for a serve_ids.py --many run (no content is printed).

usage: uv run --no-project --with jinja2 --with tokenizers python -I smoke_refusal.py RUN.json EVAL_DIR MODEL_DIR
Uses the eval harness's classifier (EVAL_DIR/lib/classify.py: split_tokens + classify_text) and MODEL_DIR/tokenizer.json.
Prints one JSON line: {"n": 5, "refusals": k, "per_prompt": [{"i", "answer_class", "reasoning_flag", "refusal", "stop"}]}
"""
import json
import sys

run_path, eval_dir, model_dir = sys.argv[1:4]
sys.path.insert(0, eval_dir)
from lib.classify import classify_text, split_tokens  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
import os  # noqa: E402

tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
res = json.load(open(run_path))
per = []
for i, r in enumerate(res["results"]):
    think, ans, started, closed = split_tokens(r["ids"])
    c = classify_text(tok.decode(think), tok.decode(ans), has_answer=started and bool(ans))
    per.append({"i": i, "answer_class": c["answer_class"], "reasoning_flag": c["reasoning_flag"],
                "refusal": c["refusal"], "stop": (r["done"] or "").split()[-1:]})
print(json.dumps({"n": len(per), "refusals": sum(p["refusal"] for p in per), "per_prompt": per}))
