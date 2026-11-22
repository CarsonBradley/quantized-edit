"""Output parsing: reasoning vs answer, native tool calls (streamed arguments), UTF-8 across tokens, stop strings."""
import json

from conftest import reply
from inkling_shim import vocab as V
from inkling_shim.parse import OutputParser, StopMatcher


def run(vocab, ids):
    p = OutputParser(vocab)
    events = []
    for t in ids:
        events += p.feed(t)
    events += p.finish()
    return p, events


def joined(events, kind):
    return "".join(e[1] for e in events if e[0] == kind)


def test_reasoning_and_answer_one_token_at_a_time(vocab):
    think = "The user wants X.\nNeed  careful   spacing.\n"
    text = " Here is the answer:\n```py\nprint('ü')\n```\n"
    p, ev = run(vocab, reply(vocab, think=think, text=text))
    assert joined(ev, "reasoning") == think
    assert joined(ev, "content") == text
    assert p.state == "done" and not p.calls


def test_text_without_header_and_no_reasoning(vocab):
    ids = vocab.encode_text("plain answer") + [V.END_SAMPLING]
    _, ev = run(vocab, ids)
    assert joined(ev, "content") == "plain answer"
    _, ev = run(vocab, [V.MSG_MODEL, V.CONTENT_THINKING, V.END_MESSAGE] + reply(vocab, text="x"))   # empty thinking
    assert joined(ev, "reasoning") == "" and joined(ev, "content") == "x"


def test_multibyte_characters_split_across_tokens(vocab):
    s = "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 ꙮ 🧑‍🔬 𓀀 é"
    ids = vocab.encode_text(s)
    assert len(ids) > len(s.split())               # some characters span several byte tokens
    _, ev = run(vocab, [V.MSG_MODEL, V.CONTENT_TEXT] + ids + [V.END_MESSAGE, V.END_SAMPLING])
    pieces = [e[1] for e in ev if e[0] == "content"]
    assert "".join(pieces) == s
    assert all("�" not in x for x in pieces)


def test_literal_tag_text_inside_answer_is_text(vocab):
    s = "write <|end_message|> and <tool_call> literally"
    _, ev = run(vocab, reply(vocab, text=s))
    assert joined(ev, "content") == s


def test_tool_call_streamed_arguments(vocab):
    args = '{"path": "src/a.py", "range": [1, 20], "note": "say \\"hi\\" }"}'
    ids = reply(vocab, think="need file", text="Reading.", calls=[("read_file", args), ("run", '{"cmd":"ls"}')])
    p, ev = run(vocab, ids)
    assert joined(ev, "content") == "Reading."
    begins = [e[1] for e in ev if e[0] == "tool_begin"]
    assert [c.name for c in begins] == ["read_file", "run"]
    for c in begins:
        streamed = "".join(e[2] for e in ev if e[0] == "tool_args" and e[1] is c)
        assert streamed == c.arguments
        assert len([e for e in ev if e[0] == "tool_args" and e[1] is c]) > 1     # really streamed in pieces
    assert begins[0].args == json.loads(args) and begins[1].args == {"cmd": "ls"}
    assert [e[0] for e in ev if e[0].startswith("tool_")][-1] == "tool_end"


def test_tool_call_spaced_head_and_malformed(vocab):
    ids = [V.MSG_MODEL] + vocab.encode_text("f") + [V.CONTENT_TOOL_JSON]
    ids += vocab.encode_text('{ "name" : "f" , "args" : {"a": 1} }') + [V.END_MESSAGE]
    ids += [V.MSG_MODEL] + vocab.encode_text("g") + [V.CONTENT_TOOL_JSON] + vocab.encode_text("oops not json")
    ids += [V.END_MESSAGE, V.END_SAMPLING]
    p, ev = run(vocab, ids)
    f, g = p.calls
    assert f.args == {"a": 1} and json.loads(f.arguments) == {"a": 1}
    assert g.name == "g" and g.args == {"raw": "oops not json"}
    assert "".join(e[2] for e in ev if e[0] == "tool_args" and e[1] is g) == g.arguments


def test_tool_text_block(vocab):
    ids = [V.MSG_MODEL] + vocab.encode_text("shell") + [V.CONTENT_TOOL_TEXT] + vocab.encode_text("ls -la")
    ids += [V.END_MESSAGE, V.END_SAMPLING]
    p, _ = run(vocab, ids)
    assert p.calls[0].name == "shell" and p.calls[0].args == {"input": "ls -la"}


def test_cut_mid_call_has_no_finished_call(vocab):
    ids = reply(vocab, calls=[("run", '{"cmd":"make test"}')], end=False)[:-4]
    p = OutputParser(vocab)
    for t in ids:
        p.feed(t)
    block, content, calls, pending = p.snapshot()
    assert block == "tool" and calls == []


def test_snapshot_of_interrupted_text(vocab):
    ids = reply(vocab, think="t", text="Hello world", end=False)[:-2]       # cut inside the answer
    p = OutputParser(vocab)
    for t in ids:
        p.feed(t)
    block, content, calls, pending = p.snapshot()
    assert block == "text" and "Hello world".startswith(content) and content and not pending


def test_stop_matcher_partial_across_chunks():
    s = StopMatcher(["</done>", "STOP"])
    out = [s.push(x) for x in ["abc</d", "o", "ne> tail"]]
    assert "".join(out) == "abc" and s.hit
    s = StopMatcher(["</done>"])
    out = [s.push(x) for x in ["a</d", "x", "y"]]
    assert "".join(out) + s.flush() == "a</dxy" and not s.hit
    s = StopMatcher("ST")
    assert s.push("xxS") == "xx" and s.flush() == "S"
    assert StopMatcher(None).push("anything") == "anything"


def test_damaged_call_head_still_parses(vocab):
    # Seen live (patched pack, temperature 0): {"name"""search_files","args":{...}}
    ids = [V.MSG_MODEL] + vocab.encode_text("search_files") + [V.CONTENT_TOOL_JSON]
    ids += vocab.encode_text('{"name"""search_files","args":{"path":"src","regex":"kv_k\\\\[v\\\\]"}}')
    ids += [V.END_MESSAGE, V.END_SAMPLING]
    p, ev = run(vocab, ids)
    c = p.calls[0]
    assert c.name == "search_files" and c.args == {"path": "src", "regex": "kv_k\\[v\\]"}
    assert json.loads(c.arguments) == c.args
