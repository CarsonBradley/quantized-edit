"""Build the fixed evaluation sets: download three public datasets (pinned revisions, sha256-checked), parse them
as data only (pyarrow) and write small JSON/JSONL files.

  uv run --no-project --with pyarrow --with tokenizers --with huggingface_hub python -I eval/prep_data.py \
      --model-dir MODEL_DIR [--out eval/data] [--raw-dir OUT/raw]

  harmful.jsonl            100 of the 104 rows of mlabonne/harmful_behaviors (test split), seed 20261122
  harmless.jsonl           100 of mlabonne/harmless_alpaca (test split), seed 20261122 (the gates use the first 30)
  wikitext2_windows.json   WikiText-2 raw test, "\n\n".join(rows), tokenized with MODEL_DIR/tokenizer.json,
                           40 evenly spaced non-overlapping windows of 512 tokens

The prompt text of these sets is not part of this repository; it is fetched here, at run time. Files already in
--raw-dir (same relative paths as on the Hub) are used as they are, after the sha256 check.
"""
import argparse, hashlib, json, os, random, sys

EVAL = os.path.dirname(os.path.abspath(__file__))
SEED = 20261122
N_HARMFUL = 100
N_HARMLESS = 100
KL_WINDOWS = 40
KL_WINDOW_LEN = 512

# (local folder, Hub dataset, revision, file, sha256)
SOURCES = {
    "harmful": ("harmful_behaviors", "mlabonne/harmful_behaviors", "01cead01398926d81f7c52bdb790ee8cf77ebba7",
                "data/test-00000-of-00001.parquet",
                "5b3d4baa7639ef78e043efe56fab40ffdcf3a38d49251db3448b3e92f802ca2a"),
    "harmless": ("harmless_alpaca", "mlabonne/harmless_alpaca", "02c6a92cfcf11bb0c387334f8146d149d65b587f",
                 "data/test-00000-of-00001.parquet",
                 "8df830a54e63adbf6e8bdf8143cb8f0cd86a9fe8dbddd21737f4cbdf22d5c08e"),
    "wikitext": ("wikitext", "Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
                 "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                 "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(raw_dir, key):
    folder, repo, rev, fn, want = SOURCES[key]
    path = os.path.join(raw_dir, folder, fn)
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        print(f"downloading {repo}@{rev[:8]} {fn}", flush=True)
        path = hf_hub_download(repo, fn, repo_type="dataset", revision=rev, local_dir=os.path.join(raw_dir, folder))
    got = sha256(path)
    if got != want:
        sys.exit(f"{path}: sha256 {got} != expected {want}")
    return path


def texts(path):
    import pyarrow.parquet as pq
    col = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    return [c for c in col if isinstance(c, str)]


def sample(rows, n, seed):
    rows = sorted(set(r.strip() for r in rows if r and r.strip()))  # dedupe + canonical order
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(rows)), min(n, len(rows))))
    return [rows[i] for i in idx], len(rows)


def write_jsonl(out, name, prompts, src, total):
    p = os.path.join(out, name)
    with open(p, "w") as f:
        for i, t in enumerate(prompts):
            f.write(json.dumps({"id": f"{name.split('.')[0]}-{i:03d}", "prompt": t}) + "\n")
    print(f"{p}: {len(prompts)} of {total} unique rows from {src} (seed {SEED})")


def main():
    ap = argparse.ArgumentParser(description="download and prepare the evaluation sets")
    ap.add_argument("--model-dir", required=True, help="folder with the model's tokenizer.json")
    ap.add_argument("--out", default=os.path.join(EVAL, "data"))
    ap.add_argument("--raw-dir", help="where the downloaded parquet files go (default OUT/raw)")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    out = os.path.abspath(a.out)
    raw = os.path.abspath(a.raw_dir or os.path.join(out, "raw"))
    os.makedirs(out, exist_ok=True)
    harm, n1 = sample(texts(fetch(raw, "harmful")), N_HARMFUL, SEED)
    write_jsonl(out, "harmful.jsonl", harm, "mlabonne/harmful_behaviors test", n1)
    safe, n2 = sample(texts(fetch(raw, "harmless")), N_HARMLESS, SEED)
    write_jsonl(out, "harmless.jsonl", safe, "mlabonne/harmless_alpaca test", n2)

    # WikiText-2 raw test -> one token stream -> evenly spaced non-overlapping windows
    stream = "\n\n".join(texts(fetch(raw, "wikitext")))
    tok = Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    ids = tok.encode(stream, add_special_tokens=False).ids
    stride = len(ids) // KL_WINDOWS
    assert stride >= KL_WINDOW_LEN
    windows = [ids[k * stride: k * stride + KL_WINDOW_LEN] for k in range(KL_WINDOWS)]
    p = os.path.join(out, "wikitext2_windows.json")
    json.dump({"source": "Salesforce/wikitext wikitext-2-raw-v1 test, '\\n\\n'.join(rows)",
               "stream_tokens": len(ids), "stride": stride, "window_len": KL_WINDOW_LEN,
               "windows": windows}, open(p, "w"))
    print(f"{p}: {KL_WINDOWS} windows x {KL_WINDOW_LEN} tokens = {KL_WINDOWS*KL_WINDOW_LEN} "
          f"(stream {len(ids)} tokens, stride {stride})")


if __name__ == "__main__":
    sys.exit(main())
