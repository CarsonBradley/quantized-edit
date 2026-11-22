"""Prefix reuse: the spliced prompt of turn N+1 must start with everything the engine fed during turn N
(prompt + generated ids but the last), even though the client drops the reasoning and re-sends plain text."""
import json

from conftest import reply
from inkling_shim import vocab as V
from inkling_shim.parse import OutputParser
from inkling_shim.prompt import ReuseMemory, Turn, normalize_messages


def turn_for(vocab, gen):
    """What service._remember stores after a generation `gen` (the engine fed gen[:-1])."""
    fed = gen[:-1]
    p = OutputParser(vocab)
    for t in fed:
        p.feed(t)
    block, content, calls, pending = p.snapshot()
    return Turn(list(fed), gen[-1] in V.STOP_IDS, block, content, calls, clean=not (block == "text" and pending))


def engine_would_reuse(fed, ids):
    return len(fed) < len(ids) and ids[:len(fed)] == fed


SYS = {"role": "system", "content": "You are a coding agent."}


def test_two_turns_with_dropped_reasoning(vocab, renderer):
    mem = ReuseMemory(vocab)
    t1 = normalize_messages([SYS, {"role": "user", "content": "hi"}])
    r1 = renderer.render(t1)
    ids1, n = mem.splice(r1)
    assert n == 0 and ids1 == r1.ids
    gen = reply(vocab, think="greet back", text=" Hello there! ")
    mem.remember(ids1, turn_for(vocab, gen))
    fed = ids1 + gen[:-1]

    # Cline sends the answer back trimmed and without reasoning_content
    t2 = normalize_messages([SYS, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello there!"},
                             {"role": "user", "content": "now fix main.py"}])
    r2 = renderer.render(t2)
    assert not engine_would_reuse(fed, r2.ids)          # plain rendering: the engine would read everything again
    ids2, n = mem.splice(r2)
    assert n == 1 and engine_would_reuse(fed, ids2)
    assert ids2 == ids1 + gen + r2.ids[r2.spans[0][1]:]

    # turn 3: the turn-2 reply is spliced too, and turn 1's splice is still applied consistently
    gen2 = reply(vocab, think="look at file", text="<read_file>\n<path>main.py</path>\n</read_file>")
    mem.remember(ids2, turn_for(vocab, gen2))
    t3 = t2 + normalize_messages([{"role": "assistant", "content": "<read_file>\n<path>main.py</path>\n</read_file>"},
                                  {"role": "user", "content": "[read_file] Result: ..."}])
    ids3, n = mem.splice(renderer.render(t3))
    assert n == 2 and engine_would_reuse(ids2 + gen2[:-1], ids3)


def test_client_appends_text(vocab, renderer):
    mem = ReuseMemory(vocab)
    ids1 = renderer.render(normalize_messages([{"role": "user", "content": "go"}])).ids
    gen = reply(vocab, think="x", text="<tool_a/>")
    mem.remember(ids1, turn_for(vocab, gen))
    note = "\n\n[Response interrupted by a tool use result.]"
    r2 = renderer.render(normalize_messages([{"role": "user", "content": "go"},
                                             {"role": "assistant", "content": "<tool_a/>" + note},
                                             {"role": "user", "content": "next"}]))
    ids2, n = mem.splice(r2)
    assert n == 1 and engine_would_reuse(ids1 + gen[:-1], ids2)
    a = len(ids1) + len(gen) - 1
    assert ids2[a:a + 2] == [V.MSG_MODEL, V.CONTENT_TEXT]
    assert vocab.decode(ids2[a + 2:a + 2 + len(vocab.encode_text(note))]) == note


def test_cancelled_mid_answer(vocab, renderer):
    """The client stopped reading mid-answer and keeps what it got (all generated tokens)."""
    mem = ReuseMemory(vocab)
    ids1 = renderer.render(normalize_messages([{"role": "user", "content": "go"}])).ids
    full = reply(vocab, think="plan", text="Step one, then step two.")
    gen = full[:-6]                                      # cancelled: the engine fed all but the last
    p = OutputParser(vocab)
    for t in gen:
        p.feed(t)
    mem.remember(ids1, turn_for(vocab, gen))
    r2 = renderer.render(normalize_messages([{"role": "user", "content": "go"},
                                             {"role": "assistant", "content": p.content},
                                             {"role": "user", "content": "continue"}]))
    ids2, n = mem.splice(r2)
    assert n == 1 and engine_would_reuse(ids1 + gen[:-1], ids2)
    assert vocab.decode(ids2[len(ids1):]).startswith(vocab.decode(gen) + "<|end_message|><|content_model_end_sampling|>")


def test_cancelled_in_reasoning(vocab, renderer):
    mem = ReuseMemory(vocab)
    ids1 = renderer.render(normalize_messages([{"role": "user", "content": "go"}])).ids
    gen = reply(vocab, think="a long chain of thought", end=False)[:-3]
    mem.remember(ids1, turn_for(vocab, gen))
    r2 = renderer.render(normalize_messages([{"role": "user", "content": "go"}, {"role": "assistant", "content": ""},
                                             {"role": "user", "content": "?"}]))
    ids2, n = mem.splice(r2)
    assert n == 1 and ids2[len(ids1):len(ids1) + len(gen) + 1] == gen[:-1] + [V.END_MESSAGE, V.END_SAMPLING]


def test_tool_call_turn_with_reformatted_arguments(vocab, renderer):
    mem = ReuseMemory(vocab)
    msgs = [{"role": "user", "content": "list files"}]
    tools = [{"type": "function", "function": {"name": "run", "parameters": {"type": "object"}}}]
    ids1 = renderer.render(normalize_messages(msgs), tools).ids
    gen = reply(vocab, think="use run", calls=[("run", '{"cmd": "ls", "all": true}')])
    mem.remember(ids1, turn_for(vocab, gen))
    back = msgs + [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "run", "arguments": json.dumps({"all": True, "cmd": "ls"})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.py"}]
    ids2, n = mem.splice(renderer.render(normalize_messages(back), tools))
    assert n == 1 and engine_would_reuse(ids1 + gen[:-1], ids2)


def test_no_splice_when_changed_or_other_conversation(vocab, renderer):
    mem = ReuseMemory(vocab)
    ids1 = renderer.render(normalize_messages([{"role": "user", "content": "A"}])).ids
    mem.remember(ids1, turn_for(vocab, reply(vocab, think="t", text="Same reply.")))
    edited = renderer.render(normalize_messages([{"role": "user", "content": "A"},
                                                 {"role": "assistant", "content": "Edited reply."},
                                                 {"role": "user", "content": "x"}]))
    assert mem.splice(edited) == (edited.ids, 0)
    other = renderer.render(normalize_messages([{"role": "user", "content": "B"},
                                                {"role": "assistant", "content": "Same reply."},
                                                {"role": "user", "content": "x"}]))
    assert mem.splice(other) == (other.ids, 0)


def test_memory_is_bounded(vocab):
    mem = ReuseMemory(vocab, capacity=3)
    for i in range(5):
        mem.remember([i], Turn([1], True, None, "", []))
    assert len(mem.turns) == 3
