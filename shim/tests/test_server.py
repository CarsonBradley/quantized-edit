"""End to end over HTTP with the fake engine: responses, streaming, tool calls, sampling keys, prefix reuse,
queueing, cancellation on disconnect, idle unload, the memory guard, crashes and shutdown."""
import http.client
import json
import os
import socket
import sys
import threading
import time

import pytest

from conftest import FAKE_ENGINE, FakeProbes, reply
from inkling_shim import vocab as V
from inkling_shim.engine import Config, Manager
from inkling_shim.memguard import GuardConfig, MemoryGuard
from inkling_shim.server import make_server
from inkling_shim.service import Service


class Harness:
    def __init__(self, server, manager, service, log_path, logs):
        self.server, self.manager, self.service, self.log_path, self.logs = server, manager, service, log_path, logs
        self.port = server.server_address[1]

    def fake_log(self):
        if not self.log_path.exists():
            return []
        return [json.loads(x) for x in self.log_path.read_text().splitlines()]

    def open_gate(self):
        if self.gate and os.path.exists(self.gate):
            os.unlink(self.gate)

    def logged(self, **match):
        return any(all(x.get(k) == v for k, v in match.items()) for x in self.fake_log())

    def generates(self):
        return [x for x in self.fake_log() if x["cmd"] == "GENERATE"]

    def post(self, body, timeout=30):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, json.loads(data) if r.getheader("Content-Type", "").startswith("application/json") else data

    def stream(self, body, timeout=30):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        c.request("POST", "/v1/chat/completions", json.dumps({**body, "stream": True}),
                  {"Content-Type": "application/json"})
        r = c.getresponse()
        assert r.status == 200, r.read()
        chunks = []
        for line in r.read().decode().split("\n\n"):
            if line.startswith("data: "):
                chunks.append(line[6:] if line == "data: [DONE]" else json.loads(line[6:]))
        c.close()
        return chunks

    def get(self, path):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("GET", path)
        r = c.getresponse()
        out = r.status, json.loads(r.read())
        c.close()
        return out


@pytest.fixture
def make(tmp_path, renderer, monkeypatch):
    made = []

    def _make(replies, *, ctx=4096, idle=0, guard=None, gate=False, **script):
        sp = tmp_path / f"script{len(made)}.json"
        if gate:
            script["gate"] = str(tmp_path / f"gate{len(made)}")
            open(script["gate"], "w").close()
        sp.write_text(json.dumps({"replies": replies, **script}))
        log_path = tmp_path / f"fake{len(made)}.log"
        monkeypatch.setenv("FAKE_SCRIPT", str(sp))
        monkeypatch.setenv("FAKE_LOG", str(log_path))
        cfg = Config(engine=[sys.executable, str(FAKE_ENGINE)], pack="fake.strata", ctx=ctx, idle=idle,
                     log_path=str(tmp_path / "engine.log"), silence_timeout=20)
        logs = []
        manager = Manager(cfg, guard=guard, log=logs.append)
        service = Service(renderer, manager, log=logs.append)
        server = make_server(service, "test-model", 0)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        h = Harness(server, manager, service, log_path, logs)
        h.gate = script.get("gate")
        made.append(h)
        return h

    yield _make
    for h in made:
        h.open_gate()
        h.server.shutdown()
        h.manager.shutdown(timeout=10)
        h.server.server_close()


def wait_for(cond, timeout=30.0):
    """Poll a condition; generous default so a loaded machine is slow, not flaky."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


MSG = [{"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "read",
                                           "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]


def test_models_and_health(make):
    h = make([[V.END_SAMPLING]])
    h.manager.cfg.extra = ["--ablate", "/tmp/DIRS.bin"]
    st, body = h.get("/v1/models")
    assert st == 200 and body["data"][0]["id"] == "test-model" and body["data"][0]["context_length"] == 4096
    meta = body["data"][0]["meta"]
    assert meta["pack"] == "fake.strata" and meta["engine_cmd"][-1].endswith("fake_engine.py")
    assert meta["engine_args"] == ["--ablate", "/tmp/DIRS.bin"]
    st, body = h.get("/health")
    assert st == 200 and body["status"] == "ok" and body["engine"] == "unloaded"
    assert body["engine_args"] == ["--ablate", "/tmp/DIRS.bin"] and body["pack"] == "fake.strata"
    assert h.manager.cfg.argv()[-2:] == ["--ablate", "/tmp/DIRS.bin"]


def test_non_stream_reasoning_and_usage(make, vocab):
    h = make([reply(vocab, think="hmm", text="Hello!")])
    st, body = h.post({"model": "x", "messages": MSG})
    assert st == 200
    msg = body["choices"][0]["message"]
    assert msg == {"role": "assistant", "content": "Hello!", "reasoning_content": "hmm"}
    assert body["choices"][0]["finish_reason"] == "stop"
    g = h.generates()[0]
    assert body["usage"]["prompt_tokens"] == len(g["ids"])
    assert body["usage"]["completion_tokens"] == len(reply(vocab, think="hmm", text="Hello!"))
    assert g["keys"]["temperature"] == "0.8" and g["keys"]["top_k"] == "40"        # llama.cpp defaults
    assert g["max_new"] == 4096 - len(g["ids"])


def test_stream_chunks(make, vocab):
    h = make([reply(vocab, think="think about it", text="Answer with ü and 😀.")])
    chunks = h.stream({"messages": MSG, "stream_options": {"include_usage": True}})
    assert chunks[-1] == "[DONE]"
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    deltas = [c["choices"][0]["delta"] for c in chunks[1:-2]]
    assert "".join(d.get("reasoning_content", "") for d in deltas) == "think about it"
    assert "".join(d.get("content", "") or "" for d in deltas) == "Answer with ü and 😀."
    assert chunks[-3]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-2]["choices"] == [] and chunks[-2]["usage"]["completion_tokens"] > 0


def test_tool_call_stream_and_plain(make, vocab):
    args = '{"path": "src/main.py"}'
    r = reply(vocab, think="need it", text="Reading the file.", calls=[("read_file", args)])
    h = make([r, r])
    chunks = h.stream({"messages": MSG, "tools": TOOLS})
    deltas = [c["choices"][0]["delta"] for c in chunks[1:-1] if c["choices"]]
    tc = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
    assert tc[0]["id"].startswith("call_") and tc[0]["function"]["name"] == "read_file" and tc[0]["type"] == "function"
    assert json.loads("".join(t["function"]["arguments"] for t in tc)) == {"path": "src/main.py"}
    assert len(tc) > 2 and all(t["index"] == 0 for t in tc)
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    st, body = h.post({"messages": MSG, "tools": TOOLS})
    msg = body["choices"][0]["message"]
    assert msg["content"] == "Reading the file." and body["choices"][0]["finish_reason"] == "tool_calls"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"path": "src/main.py"}


def test_sampling_keys_max_tokens_and_length(make, vocab):
    h = make([reply(vocab, text="one two three four five six seven")])
    st, body = h.post({"messages": MSG, "temperature": 0, "top_p": 0.5, "top_k": 0, "min_p": 0.1, "seed": 7,
                       "presence_penalty": 0.5, "frequency_penalty": 0.25, "max_tokens": 5})
    g = h.generates()[0]
    assert g["max_new"] == 5
    assert g["keys"] == {"temperature": "0.0", "top_p": "0.5", "top_k": "128", "min_p": "0.1", "seed": "7",
                         "penalty_present": "0.5", "penalty_freq": "0.25"}
    assert body["choices"][0]["finish_reason"] == "length" and body["usage"]["completion_tokens"] == 5


def test_stop_string_cancels_engine(make, vocab):
    h = make([reply(vocab, text="keep this</done> drop this and much more text after it")], token_delay=0.02)
    st, body = h.post({"messages": MSG, "stop": ["</done>"]})
    assert body["choices"][0]["message"]["content"] == "keep this"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert wait_for(lambda: h.logged(cmd="CANCEL"))


def test_context_overflow_is_400(make):
    h = make([[V.END_SAMPLING]], ctx=64)
    st, body = h.post({"messages": [{"role": "user", "content": "word " * 200}]})
    assert st == 400 and body["error"]["code"] == "context_length_exceeded" and body["error"]["n_ctx"] == 64
    assert h.manager.engine is None                       # rejected before anything was loaded


def test_bad_requests(make):
    h = make([[V.END_SAMPLING]])
    assert h.post({"messages": []})[0] == 400
    assert h.post({"messages": MSG, "n": 2})[0] == 400
    assert h.post({"messages": MSG, "reasoning_effort": "huge"})[0] == 400
    assert h.post({"messages": MSG, "temperature": "hot"})[0] == 400


def test_prefix_reuse_across_turns(make, vocab):
    """Turn 2 is rendered without the reasoning (as Cline sends it); the engine still resumes after turn 1."""
    gen1 = reply(vocab, think="greeting", text="Hello!")
    h = make([gen1, reply(vocab, text="Done.")])
    st, b1 = h.post({"messages": [{"role": "system", "content": "sys " * 300}, *MSG]})
    turn2 = [{"role": "system", "content": "sys " * 300}, *MSG, {"role": "assistant", "content": "Hello!"},
             {"role": "user", "content": "next"}]
    st, b2 = h.post({"messages": turn2})
    g1, g2 = h.generates()
    assert g2["ids"][:len(g1["ids"]) + len(gen1) - 1] == g1["ids"] + gen1[:-1]
    assert g2["reuse"] == len(g1["ids"]) + len(gen1) - 1
    assert b2["usage"]["prompt_tokens_details"]["cached_tokens"] == g2["reuse"]
    assert b2["timings"]["spliced_turns"] == 1


def test_reuse_off_rereads(make, vocab, renderer):
    gen1 = reply(vocab, think="greeting", text="Hello!")
    h = make([gen1, reply(vocab, text="Done.")])
    h.service.reuse = False
    h.post({"messages": MSG})
    h.post({"messages": [*MSG, {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "next"}]})
    assert h.generates()[1]["reuse"] == 0


def test_requests_queue_in_order(make, vocab):
    h = make([reply(vocab, text="first"), reply(vocab, text="second"), reply(vocab, text="third")], gate=True)
    out = {}

    def go(i):
        out[i] = h.post({"messages": [{"role": "user", "content": f"q{i}"}]}, timeout=60)[1]
    threads = []
    for i in range(3):
        t = threading.Thread(target=go, args=(i,))
        t.start()
        threads.append(t)
        if i == 0:                                         # the first one holds the engine (gate closed)
            assert wait_for(lambda: h.logged(cmd="GENERATE"))
        else:                                              # the others are queued, in this order
            assert wait_for(lambda: h.manager.lock.waiting == i)
    h.open_gate()
    for t in threads:
        t.join(60)
    assert [out[i]["choices"][0]["message"]["content"] for i in range(3)] == ["first", "second", "third"]
    assert len({x["pid"] for x in h.fake_log() if x["cmd"] == "START"}) == 1


def raw_request(port, body):
    s = socket.create_connection(("127.0.0.1", port))
    data = json.dumps(body).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data)
    return s


def test_disconnect_mid_stream_cancels(make, vocab):
    long = reply(vocab, think="x " * 50, text="y " * 300)
    h = make([long, reply(vocab, text="after")], token_delay=0.05)
    s = raw_request(h.port, {"messages": MSG, "stream": True})
    s.settimeout(30)
    got = b""
    while got.count(b"data: ") < 5:
        got += s.recv(4096)
    s.close()
    assert wait_for(lambda: h.logged(cmd="DONE", reason="cancel"))
    st, body = h.post({"messages": MSG})                 # the engine is free and in step again
    assert body["choices"][0]["message"]["content"] == "after"


def test_disconnect_during_prefill_cancels(make, vocab):
    h = make([reply(vocab, text="never sent"), reply(vocab, text="ok")], gate=True)   # held before the first chunk
    s = raw_request(h.port, {"messages": [{"role": "user", "content": "long " * 3000}]})
    assert wait_for(lambda: h.logged(cmd="GENERATE"))
    s.close()
    assert wait_for(lambda: h.logged(cmd="DONE", reason="cancel", gen=0))
    h.open_gate()
    assert h.post({"messages": MSG})[1]["choices"][0]["message"]["content"] == "ok"


def test_client_leaving_queue_is_dropped(make, vocab):
    h = make([reply(vocab, text="slow"), reply(vocab, text="third")], gate=True)
    t = threading.Thread(target=h.post, args=({"messages": MSG}, 60))
    t.start()
    assert wait_for(lambda: h.logged(cmd="GENERATE"))
    s = raw_request(h.port, {"messages": [{"role": "user", "content": "queued then gone"}]})
    assert wait_for(lambda: h.manager.lock.waiting == 1)
    s.close()
    assert wait_for(lambda: h.manager.lock.waiting == 0)   # it gave up its place while the gate is still closed
    h.open_gate()
    t.join(60)
    h.post({"messages": MSG})
    assert len(h.generates()) == 2                        # the abandoned request never reached the engine


def test_idle_unload_and_reload(make, vocab):
    h = make([reply(vocab, text="a")], idle=0.6)
    h.post({"messages": MSG})
    proc = h.manager.engine.proc
    assert proc.poll() is None
    # _unload clears manager.engine first, waits for the process, then logs: wait for all three, not a fixed time
    assert wait_for(lambda: h.manager.engine is None and proc.poll() is not None
                    and any("engine stopped (idle" in x for x in h.logs))
    assert h.logged(cmd="QUIT")                            # it was asked to quit, not killed
    h.post({"messages": MSG})
    assert h.manager.loads == 2 and h.manager.engine.proc is not proc


def test_guard_refuses_to_load(make):
    guard = MemoryGuard(18.7, GuardConfig(poll_s=0.1, health_url="http://127.0.0.1:9/health", total_gb=64.0),
                        FakeProbes(procs={"llama-server": {4242}}))   # another model while the service is up
    h = make([[V.END_SAMPLING]], guard=guard)
    st, body = h.post({"messages": MSG})
    assert st == 503 and "llama-server pid [4242]" in body["error"]["message"]
    assert h.manager.engine is None and not h.fake_log()
    st, _ = h.post({"messages": MSG, "stream": True})
    assert st == 503                                       # streaming requests get a real HTTP error too


def test_engine_crash_then_recovery(make, vocab):
    h = make([reply(vocab, text="crash soon " * 10)], crash_after=3)
    st, body = h.post({"messages": MSG})
    assert st == 500 and "exited" in body["error"]["message"]
    assert h.manager.engine is None or not h.manager.engine.alive()     # reaped before the next request
    os.environ["FAKE_SCRIPT"] = str(h.log_path.parent / "ok.json")
    (h.log_path.parent / "ok.json").write_text(json.dumps({"replies": [reply(vocab, text="back")]}))
    st, body = h.post({"messages": MSG})
    assert st == 200 and body["choices"][0]["message"]["content"] == "back" and h.manager.loads == 2


def test_load_failure_is_500(make):
    h = make([[V.END_SAMPLING]], fail_load=True)
    st, body = h.post({"messages": MSG})
    assert st == 500 and h.manager.state == "unloaded"


def test_shutdown_mid_generation_leaves_no_engine(make, vocab):
    h = make([reply(vocab, text="z " * 50)], gate=True)
    out = []
    t = threading.Thread(target=lambda: out.append(h.post({"messages": MSG}, 60)))
    t.start()
    assert wait_for(lambda: h.logged(cmd="GENERATE"))      # mid-request, held by the gate
    proc = h.manager.engine.proc
    h.manager.shutdown(timeout=30)
    t.join(60)
    assert out and out[0][0] == 500 and "shutting down" in out[0][1]["error"]["message"]   # the client is told
    assert wait_for(lambda: proc.poll() is not None)
    assert h.logged(cmd="CANCEL") and h.logged(cmd="EXIT")


def test_unload_endpoint(make, vocab):
    h = make([reply(vocab, text="a")])
    h.post({"messages": MSG})
    proc = h.manager.engine.proc
    c = http.client.HTTPConnection("127.0.0.1", h.port, timeout=60)
    c.request("POST", "/unload", b"")
    assert json.loads(c.getresponse().read()) == {"unloaded": True}
    assert wait_for(lambda: proc.poll() is not None)


def test_images_become_a_note(make, vocab):
    h = make([reply(vocab, text="ok")])
    st, _ = h.post({"messages": [{"role": "user", "content": [{"type": "text", "text": "see"},
                                                              {"type": "image_url", "image_url": {"url": "x"}}]}]})
    assert st == 200 and "image omitted" in vocab.decode(h.generates()[0]["ids"])


def test_engine_args_cannot_add_slots():
    for extra in (["--parallel", "4"], ["--batch", "2"], ["--slots", "2"]):
        with pytest.raises(ValueError):
            Config(engine=["x"], pack="p", extra=extra).argv()
    argv = Config(engine=["x"], pack="p", extra=["--ablate", "D.bin"]).argv()
    assert argv[argv.index("--parallel") + 1] == "1" and argv.count("--parallel") == 1


def test_undeclared_native_call_becomes_cline_xml_and_splices(make, vocab):
    """Cline sends no `tools`; the model may still answer with a native call. It reaches Cline as its XML format,
    and the next turn (with that text sent back) still reuses the engine's cache."""
    gen = reply(vocab, think="search it", text="Let me search.",
                calls=[("search_files", '{"path":"engine","regex":"kv_k|kv_v"}')])
    h = make([gen, reply(vocab, text="<attempt_completion>\n<result>done</result>\n</attempt_completion>")])
    chunks = h.stream({"messages": MSG})
    content = "".join((c["choices"][0]["delta"].get("content") or "") for c in chunks if isinstance(c, dict)
                      and c.get("choices"))
    assert content == "Let me search.\n\n<search_files>\n<path>engine</path>\n<regex>kv_k|kv_v</regex>\n</search_files>"
    assert not any("tool_calls" in c["choices"][0]["delta"] for c in chunks if isinstance(c, dict) and c.get("choices"))
    st, body = h.post({"messages": [*MSG, {"role": "assistant", "content": content},
                                    {"role": "user", "content": "[search_files] Result: none"}]})
    g1, g2 = h.generates()
    assert g2["reuse"] == len(g1["ids"]) + len(gen) - 1 and body["timings"]["spliced_turns"] == 1
