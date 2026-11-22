"""Live checks against a running inkling-s with the real engine (GPU). Not part of the pytest suite.
Standard library only; runs with /usr/bin/python3 (3.9).

    python3 tests/live_check.py [--url http://127.0.0.1:8097] [--idle 300] STEP [STEP ...]
    steps: memory short tool long reuse cline cancel cancel-prefill idle

Each step prints its measurements and PASS / FAIL lines (the expected numbers are in shim/README.md).
Some checks read the shim's own log (run/shim.log) and the engine's stderr (run/engine.log). The prompts come from
the pack's Hugging Face folder: set INKLING_S_MODEL_DIR (its tools/prompts/ is used).
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

SHIM = Path(__file__).resolve().parent.parent
PROMPTS = Path(os.environ.get("INKLING_S_MODEL_DIR", ".")) / "tools" / "prompts"
SHIM_LOG = SHIM / "run" / "shim.log"
ENGINE_LOG = SHIM / "run" / "engine.log"
FAILS = []


def check(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def log_tail(path, since_bytes):
    try:
        with open(path, "rb") as f:
            f.seek(since_bytes)
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def log_size(path):
    try:
        return path.stat().st_size
    except OSError:
        return 0


class Client:
    def __init__(self, url):
        u = urlparse(url)
        self.host, self.port = u.hostname, u.port

    def conn(self, timeout=900):
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def get(self, path):
        c = self.conn(10)
        c.request("GET", path)
        return json.loads(c.getresponse().read())

    def health(self):
        return self.get("/health")

    def post(self, body):
        c = self.conn()
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, json.loads(r.read())

    def stream(self, body, stop_after=None):
        """-> (chunks, seconds to first token). stop_after=N: drop the connection after N chunks."""
        c = self.conn()
        t0 = time.monotonic()
        c.request("POST", "/v1/chat/completions",
                  json.dumps(dict(body, stream=True, stream_options={"include_usage": True})),
                  {"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            raise SystemExit(f"HTTP {r.status}: {r.read()[:2000]!r}")
        chunks, ttft = [], None
        while True:
            line = r.fp.readline()
            if not line:
                break
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            obj = json.loads(data)
            chunks.append(obj)
            d = (obj.get("choices") or [{}])[0].get("delta") or {}
            if ttft is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                ttft = time.monotonic() - t0
            if stop_after and len(chunks) >= stop_after:
                r.fp.raw._sock.shutdown(socket.SHUT_RDWR)   # the response owns the socket by now
                r.close()
                c.close()
                return chunks, ttft
        c.close()
        return chunks, ttft


def collect(chunks):
    out = {"reasoning": "", "content": "", "calls": {}, "finish": None, "usage": None, "timings": None,
           "error": None}
    for ch in chunks:
        if ch.get("error"):
            out["error"] = ch["error"]
        if ch.get("usage"):
            out["usage"] = ch["usage"]
        for choice in ch.get("choices") or []:
            d = choice.get("delta") or {}
            out["reasoning"] += d.get("reasoning_content") or ""
            out["content"] += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                slot = out["calls"].setdefault(tc["index"], {"name": None, "arguments": ""})
                slot["name"] = (tc.get("function") or {}).get("name") or slot["name"]
                slot["arguments"] += (tc.get("function") or {}).get("arguments") or ""
            if choice.get("finish_reason"):
                out["finish"] = choice["finish_reason"]
                out["timings"] = ch.get("timings")
    return out


def show(name, r, ttft=None, chars=400):
    print(f"[{name}] finish={r['finish']} usage={r['usage']} ttft={ttft and round(ttft, 2)}s error={r['error']}")
    print(f"[{name}] timings={json.dumps(r['timings'])}")
    print(f"[{name}] reasoning[:{chars}]={r['reasoning'][:chars]!r}")
    print(f"[{name}] content[:{chars}]={r['content'][:chars]!r}")


TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a text file from the user's project and return its contents.",
    "parameters": {"type": "object", "required": ["path"],
                   "properties": {"path": {"type": "string", "description": "path relative to the project root"}}}}}]


def step_memory(c, a):
    """Load the engine (one tiny request) and report TotalAlloc from the engine log."""
    before = log_size(ENGINE_LOG)
    h = c.health()
    print(f"[memory] ctx {h['ctx']}, binary {' '.join(h['engine_cmd'])}, pack {h['pack']}, args {h['engine_args']}")
    if h["engine"] == "loaded":
        print("[memory] engine already loaded; run `inkling-s unload` first to see its allocation line")
    st, body = c.post({"messages": [{"role": "user", "content": "Say OK."}], "reasoning_effort": "none",
                       "max_tokens": 8, "temperature": 0})
    check(st == 200, "memory: tiny request served")
    m = re.findall(r"runner ready: .*", log_tail(ENGINE_LOG, before))
    print("[memory] " + (m[-1] if m else "no 'runner ready' line in run/engine.log"))
    alloc = re.search(r"TotalAlloc=([\d.]+) GiB", m[-1]) if m else None
    if alloc:
        expect = 15.46 + h["ctx"] / 32768 + (0.44 * h["ctx"] / 32768 if h["draft"] != "off" else 0)
        got = float(alloc.group(1))
        check(abs(got - expect) < 0.6, f"memory: TotalAlloc {got} GiB vs predicted {expect:.2f} GiB")


def step_short(c, a):
    body = {"messages": [{"role": "user", "content": (PROMPTS / "p_short.txt").read_text()}], "max_tokens": 1024,
            "temperature": 0}
    chunks, ttft = c.stream(body)
    r = collect(chunks)
    show("short", r, ttft)
    check(r["usage"]["prompt_tokens"] == 63, "short: prompt is 63 tokens (same as the pack's tools/strata_real.py)")
    check(bool(r["reasoning"]), "short: reasoning arrives as reasoning_content")
    check(bool(r["content"]) or r["finish"] == "length", "short: answer arrives as content")
    check(r["finish"] in ("stop", "length"), "short: finish_reason stop/length")
    check("<|" not in r["content"] + r["reasoning"], "short: no raw special tokens in the text")


def step_tool(c, a):
    before = log_size(SHIM_LOG)
    msgs = [{"role": "system", "content": "You are a coding assistant. Use the tools to look at files."},
            {"role": "user", "content": "Open src/app.py with the read_file tool and tell me what it does."}]
    st, body = c.post({"messages": msgs, "tools": TOOLS, "temperature": 0, "max_tokens": 2048})
    print("[tool] turn 1: " + json.dumps(body)[:1500])
    raw = re.findall(r"raw output: (.*)", log_tail(SHIM_LOG, before))
    print("[tool] raw model output (needs INKLING_S_DEBUG=1): " + (raw[-1][:1500] if raw else "(debug off)"))
    msg = body["choices"][0]["message"] if st == 200 else {}
    calls = msg.get("tool_calls") or []
    check(st == 200 and body["choices"][0]["finish_reason"] == "tool_calls", "tool: finish_reason tool_calls")
    check(bool(calls) and calls[0]["function"]["name"] == "read_file", "tool: calls read_file")
    try:
        args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    except ValueError:
        args = {}
    check("path" in args, "tool: arguments are JSON with a path")
    if raw:
        check("<|content_invoke_tool_json|>" in raw[-1] and '"args"' in raw[-1],
              "tool: the model writes the template's native format (<|content_invoke_tool_json|>{name, args})")
    if not calls:
        return
    msgs += [{"role": "assistant", "content": msg.get("content") or None, "tool_calls": calls},
             {"role": "tool", "tool_call_id": calls[0]["id"],
              "content": "import sys\n\ndef main():\n    print('hello', sys.argv[1:])\n\nif __name__ == '__main__':\n"
                         "    main()\n"}]
    chunks, ttft = c.stream({"messages": msgs, "tools": TOOLS, "temperature": 0, "max_tokens": 2048})
    r = collect(chunks)
    show("tool turn 2", r, ttft)
    t = r["timings"] or {}
    check(r["finish"] == "stop" and bool(r["content"]), "tool: turn 2 answers in text")
    check(t.get("spliced_turns") == 1 and t.get("cache_n", 0) > 0,
          "tool: turn 2 reused the KV prefix (spliced_turns 1, cache_n > 0)")
    chunks, _ = c.stream({"messages": msgs[:2], "tools": TOOLS, "temperature": 0, "max_tokens": 2048})
    r = collect(chunks)
    try:
        ok = bool(r["calls"]) and "path" in json.loads(r["calls"][0]["arguments"])
    except ValueError:
        ok = False
    check(r["finish"] == "tool_calls" and ok, "tool: streamed call arguments reassemble to JSON")


def step_long(c, a):
    body = {"messages": [{"role": "user", "content": (PROMPTS / "p_28k.txt").read_text()}], "max_tokens": 512,
            "temperature": 0}
    chunks, ttft = c.stream(body)
    r = collect(chunks)
    show("long", r, ttft)
    t = r["timings"] or {}
    check(r["usage"]["prompt_tokens"] == 28156, "long: prompt is 28,156 tokens")
    pps = (t.get("prompt_n") or 0) / max(1e-9, (t.get("prompt_ms") or 0) / 1000)
    check(pps >= 700, f"long: prompt read at >= 700 tok/s (got {pps:.0f}, cache_n {t.get('cache_n')})")
    check((t.get("predicted_per_second") or 0) >= 20, "long: decode >= 20 tok/s")


def cline_start():
    return [{"role": "system", "content": (PROMPTS / "p_28k.txt").read_text()},
            {"role": "user", "content": [
                {"type": "text", "text": "<task>\nIn two sentences, what is this code for?\n</task>"},
                {"type": "text", "text": "<environment_details>\n# Current Time\n2026-11-22\n</environment_details>"}]}]


def step_reuse(c, a):
    """Three Cline-shaped turns at ~28K: turns 2 and 3 must read only their new tokens."""
    msgs = cline_start()
    t0 = time.monotonic()
    chunks, ttft = c.stream({"messages": msgs, "temperature": 0, "max_tokens": 1500})
    prev = collect(chunks)
    show("reuse turn 1", prev, ttft, 1200)
    print(f"[reuse turn 1] wall {time.monotonic() - t0:.1f}s")
    for turn, ask in ((2, "Now name three functions or classes from it, one line each."),
                      (3, "Which of those would you test first, and why? Two sentences.")):
        msgs = msgs + [{"role": "assistant", "content": prev["content"]},     # Cline drops reasoning_content
                       {"role": "user", "content": [{"type": "text", "text": ask}]}]
        t0 = time.monotonic()
        chunks, ttft = c.stream({"messages": msgs, "temperature": 0, "max_tokens": 1500})
        r = collect(chunks)
        show(f"reuse turn {turn}", r, ttft, 1200)
        t = r["timings"] or {}
        want = prev["usage"]["prompt_tokens"] + prev["usage"]["completion_tokens"] - 1
        check(t.get("cache_n") == want, f"reuse turn {turn}: cache_n {t.get('cache_n')} == previous prompt + "
                                        f"completion - 1 ({want})")
        check((t.get("prompt_n") or 1e9) < 500 and (t.get("prompt_ms") or 1e9) < 3000,
              f"reuse turn {turn}: only the new tokens are read ({t.get('prompt_n')} tokens, "
              f"{t.get('prompt_ms')} ms)")
        check(bool(r["content"].strip()) and r["finish"] == "stop", f"reuse turn {turn}: a complete answer")
        print(f"[reuse turn {turn}] wall {time.monotonic() - t0:.1f}s  <- read the answers above: coherent, on topic,"
              f" consistent with earlier turns?")
        prev = r


CLINE_SYSTEM = """You are Cline, a highly skilled software engineer with extensive knowledge in many programming languages, frameworks, design patterns, and best practices.

====

TOOL USE

You have access to a set of tools that are executed upon the user's approval. You can use one tool per message, and will receive the result of that tool use in the user's response. You use tools step-by-step to accomplish a given task, with each tool use informed by the result of the previous tool use.

# Tool Use Formatting

Tool use is formatted using XML-style tags. The tool name is enclosed in opening and closing tags, and each parameter is similarly enclosed within its own set of tags. Here's the structure:

<tool_name>
<parameter1_name>value1</parameter1_name>
<parameter2_name>value2</parameter2_name>
</tool_name>

Always adhere to this format for the tool use to ensure proper parsing and execution.

# Tools

## execute_command
Description: Request to execute a CLI command on the system.
Parameters:
- command: (required) The CLI command to execute.
- requires_approval: (required) true or false.
Usage:
<execute_command>
<command>Your command here</command>
<requires_approval>true or false</requires_approval>
</execute_command>

## read_file
Description: Request to read the contents of a file at the specified path.
Parameters:
- path: (required) The path of the file to read (relative to the current working directory).
Usage:
<read_file>
<path>File path here</path>
</read_file>

## search_files
Description: Request to perform a regex search across files in a specified directory.
Parameters:
- path: (required) The directory to search in.
- regex: (required) The regular expression pattern to search for.
Usage:
<search_files>
<path>Directory path here</path>
<regex>Your regex pattern here</regex>
</search_files>

## attempt_completion
Description: Once you have completed the task, use this tool to present the result to the user.
Parameters:
- result: (required) The result of the task.
Usage:
<attempt_completion>
<result>
Your final result description here
</result>
</attempt_completion>

# Tool Use Guidelines

1. Choose the most appropriate tool based on the task and the tool descriptions provided.
2. Use one tool at a time per message, and wait for the result before continuing.
3. Formulate your tool use using the XML format specified for each tool.

====

RULES

- Your current working directory is: /home/dev/strata
- Do not ask for more information than necessary. Use the tools provided to accomplish the user's request efficiently.
- When you have finished the task, you must use the attempt_completion tool.

====

PROJECT CONTEXT

The user has this file open in the editor: engine/src/inkling2/metal/ink_metal_runner.h. An excerpt follows.

"""


def step_cline(c, a):
    """Mimics Cline: ~28K system prompt with XML tool instructions in plain text, streamed, two turns, no `tools`."""
    sys_prompt = CLINE_SYSTEM + (PROMPTS / "p_28k.txt").read_text()
    env = "<environment_details>\n# VSCode Visible Files\nengine/src/inkling2/metal/ink_metal_runner.h\n\n" \
          "# Current Working Directory (/home/dev/strata) Files\nengine/\ntools/\nREADME.md\n</environment_details>"
    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "<task>\nFind where the KV cache buffers are allocated in "
                                         "engine/src/inkling2/metal/ink_metal_runner.h and tell me how many bytes "
                                         "they take per token. Read the file first.\n</task>"},
                {"type": "text", "text": env}]}]
    tags = re.compile(r"<(read_file|search_files|execute_command|attempt_completion)>")
    t0 = time.monotonic()
    chunks, ttft = c.stream({"model": "inkling-2-mini-strata", "messages": msgs, "temperature": 0,
                             "max_tokens": 8192})
    r1 = collect(chunks)
    show("cline turn 1", r1, ttft, 1500)
    print(f"[cline turn 1] wall {time.monotonic() - t0:.1f}s, {len(chunks)} chunks")
    check(r1["usage"]["prompt_tokens"] > 27000, f"cline: system prompt is ~28K tokens ({r1['usage']['prompt_tokens']})")
    check(len(chunks) > 20, "cline: the reply streamed in many chunks")
    check(not r1["calls"] and r1["finish"] == "stop", "cline: plain text reply, no native tool_calls, finish stop")
    m = tags.search(r1["content"])
    check(bool(m), "cline: the reply uses a Cline XML tool (" + (m.group(1) if m else "none") + ")")
    check("<|" not in r1["content"], "cline: no special-token text leaked")
    result = ("[read_file for 'engine/src/inkling2/metal/ink_metal_runner.h'] Result:\n"
              "        for (int v = 0; v < c.n_visits; ++v) {\n"
              "            const int ip = c.phys_for_visit(v);\n"
              "            const int stride = c.is_swa[ip] ? c.swa_win : max_ctx;\n"
              "            const size_t kvb = (size_t)max_slots * stride * c.n_kv * c.head_dim * sizeof(uint16_t);\n"
              "            kv_k[v] = cx->alloc(kvb);\n"
              "            kv_v[v] = cx->alloc(kvb);\n")
    msgs = msgs + [{"role": "assistant", "content": r1["content"]},
                   {"role": "user", "content": [{"type": "text", "text": result}, {"type": "text", "text": env}]}]
    t0 = time.monotonic()
    chunks, ttft = c.stream({"model": "inkling-2-mini-strata", "messages": msgs, "temperature": 0,
                             "max_tokens": 8192})
    r2 = collect(chunks)
    show("cline turn 2", r2, ttft, 1500)
    t = r2["timings"] or {}
    want = r1["usage"]["prompt_tokens"] + r1["usage"]["completion_tokens"] - 1
    print(f"[cline turn 2] wall {time.monotonic() - t0:.1f}s, ttft {ttft and round(ttft, 2)}s")
    check(t.get("cache_n") == want, f"cline turn 2: reused the whole previous turn (cache_n {t.get('cache_n')} == {want})")
    check((t.get("prompt_n") or 1e9) < 1000, f"cline turn 2: re-read only the new tokens ({t.get('prompt_n')})")
    check(r2["finish"] == "stop" and bool(tags.search(r2["content"])), "cline turn 2: answers with a Cline XML tool")


def wait_free(c, timeout):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if not c.health()["busy"]:
            return time.monotonic() - t0
        time.sleep(0.1)
    return None


def cancel_latencies(text):
    return [int(x) for x in re.findall(r"cancel latency (\d+) ms", text)]


def step_cancel(c, a):
    before = log_size(SHIM_LOG)
    body = {"messages": [{"role": "user", "content": "Write a 3000-word essay on the history of the printing press."}],
            "temperature": 0, "max_tokens": 6000}
    chunks, _ = c.stream(body, stop_after=40)
    print(f"[cancel] dropped the connection after {len(chunks)} chunks")
    free = wait_free(c, 10)
    check(free is not None, f"cancel: engine free again within 10 s (took {free and round(free, 2)} s)")
    lat = cancel_latencies(log_tail(SHIM_LOG, before))
    check(bool(lat), f"cancel: shim logged the CANCEL -> DONE latency {lat} ms")
    t0 = time.monotonic()
    st, _ = c.post({"messages": [{"role": "user", "content": "Say OK."}], "reasoning_effort": "none",
                    "max_tokens": 20, "temperature": 0})
    check(st == 200, f"cancel: next request served ({time.monotonic() - t0:.1f}s)")


def step_cancel_prefill(c, a):
    before = log_size(SHIM_LOG)
    body = {"messages": [{"role": "user", "content": (PROMPTS / "p_28k.txt").read_text() + "\n(cancel test)"}],
            "max_tokens": 64, "stream": True}
    conn = c.conn()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    time.sleep(8)                                            # mid-way through a ~35 s prompt read
    conn.sock.shutdown(socket.SHUT_RDWR)
    conn.close()
    free = wait_free(c, 5)
    check(free is not None, f"cancel-prefill: engine stops reading within 5 s (took {free and round(free, 2)} s)")
    print(f"[cancel-prefill] cancel latency {cancel_latencies(log_tail(SHIM_LOG, before))} ms")


def step_idle(c, a):
    h = c.health()
    if h["engine"] != "loaded":
        c.post({"messages": [{"role": "user", "content": "Say OK."}], "reasoning_effort": "none", "max_tokens": 10})
        h = c.health()
    pid, loads = h["engine_pid"], h["loads"]
    print(f"[idle] engine pid {pid}; waiting {a.idle + 15:.0f}s for the idle unload")
    time.sleep(a.idle + 15)
    h = c.health()
    try:
        os.kill(pid, 0)
        gone = False
    except (ProcessLookupError, TypeError):
        gone = True
    check(h["engine"] == "unloaded" and gone, "idle: engine unloaded and its process gone")
    st, _ = c.post({"messages": [{"role": "user", "content": "Say OK."}], "reasoning_effort": "none", "max_tokens": 10})
    check(st == 200 and c.health()["loads"] == loads + 1, "idle: the next request loads it again")


STEPS = {"memory": step_memory, "short": step_short, "tool": step_tool, "long": step_long, "reuse": step_reuse,
         "cline": step_cline,
         "cancel": step_cancel, "cancel-prefill": step_cancel_prefill, "idle": step_idle}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8097")
    p.add_argument("--idle", type=float, default=float(os.environ.get("INKLING_S_IDLE", 300)))
    p.add_argument("steps", nargs="+", choices=list(STEPS))
    a = p.parse_args()
    c = Client(a.url)
    for s in a.steps:
        print(f"===== {s}", flush=True)
        t0 = time.monotonic()
        STEPS[s](c, a)
        print(f"===== {s} done in {time.monotonic() - t0:.1f}s", flush=True)
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
