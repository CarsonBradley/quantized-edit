"""Renderer parity: chat_template.jinja through our Jinja environment vs transformers' apply_chat_template,
character for character and token for token, plus tokenization vs plain `tokenizers`."""
import json

import pytest

from conftest import MODEL_DIR, need_model_dir
from inkling_shim import vocab as V
from inkling_shim.prompt import BadRequest, normalize_messages, resolve_effort

transformers = pytest.importorskip("transformers")

# A ~28K-token Cline-style system prompt; it ships with the pack's Hugging Face folder (tools/prompts/).
_P28K_PATH = MODEL_DIR / "tools" / "prompts" / "p_28k.txt" if MODEL_DIR else None
P28K = _P28K_PATH.read_text(encoding="utf-8") if _P28K_PATH and _P28K_PATH.is_file() else None

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file. Use <path> & \"quotes\"; returns 'text' > 0 bytes.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "relative path"},
            "range": {"type": "array", "items": {"type": "integer"}}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "run", "description": "Run a shell command",
                                      "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}},
    {"name": "bare_tool", "description": "", "parameters": {}},
]

SYS = "You are Cline, a highly skilled software engineer.\n\n====\n\nTOOL USE\n\n<read_file>\n<path>x</path>\n</read_file>"

CASES = {
    "user_only": ([{"role": "user", "content": "Hello"}], None, None),
    "system_user": ([{"role": "system", "content": SYS}, {"role": "user", "content": "Fix the bug."}], None, None),
    "multi_turn": ([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "2+2?"},
                    {"role": "assistant", "content": "4"}, {"role": "user", "content": "times 3?"}], None, None),
    "history_reasoning": ([{"role": "user", "content": "q"},
                           {"role": "assistant", "content": "a", "reasoning_content": "let me think\nok"},
                           {"role": "user", "content": "more"}], None, None),
    "tools_full_cycle": ([
        {"role": "system", "content": "Use tools."}, {"role": "user", "content": "Read main.py"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "read_file",
                                                              "arguments": '{"path": "main.py", "range": [1, 20]}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "print('hi')\n"},
        {"role": "assistant", "content": "It prints hi."}, {"role": "user", "content": "thanks"}], TOOLS, None),
    "tool_named_and_parallel": ([
        {"role": "user", "content": "do two things"},
        {"role": "assistant", "content": "Running both.", "reasoning_content": "two calls", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "run", "arguments": {"cmd": "ls -la"}}},
            {"id": "b", "type": "function", "function": {"name": "read_file", "arguments": json.dumps(
                {"z": None, "a": True, "m": 1.5, "nested": {"k": ["x", 2, {"y": "ü"}]}, "path": "a b.txt"})}}]},
        {"role": "tool", "name": "run", "content": "total 0"},
        {"role": "tool", "tool_call_id": "b", "content": [{"type": "text", "text": "file body"}]}], TOOLS, None),
    "content_parts": ([{"role": "system", "content": [{"type": "text", "text": "sys part"}]},
                       {"role": "user", "content": [{"type": "text", "text": "<task>\nfix it\n</task>"},
                                                    {"type": "text", "text": "<environment_details>\n# cwd\n/tmp\n"
                                                                             "</environment_details>"}]}], None, None),
    "unicode_whitespace": ([{"role": "user", "content": "naïve café 😀 日本語 שלום \t tab\r\nCRLF   trailing   \n\n\n"}],
                           None, None),
    "only_system": ([{"role": "system", "content": "just a system prompt"}], None, None),
    "empty_contents": ([{"role": "user", "content": ""}, {"role": "assistant", "content": ""},
                        {"role": "user", "content": "again"}], None, None),
    "assistant_name": ([{"role": "user", "content": "hi"}, {"role": "assistant", "name": "run", "content": "x",
                                                            "tool_calls": [{"id": "1", "type": "function",
                                                                            "function": {"name": "run",
                                                                                         "arguments": "{}"}}]}],
                       TOOLS, None),
    "cline_28k": ([{"role": "system", "content": P28K or ""},
                   {"role": "user", "content": [{"type": "text", "text": "<task>\nRefactor the parser.\n</task>"},
                                                {"type": "text", "text": "<environment_details>\nVSCode Visible Files\n"
                                                                         "parse.py\n</environment_details>"}]},
                   {"role": "assistant", "content": "I'll read it first.\n\n<read_file>\n<path>parse.py</path>\n"
                                                    "</read_file>"},
                   {"role": "user", "content": [{"type": "text", "text": "[read_file for 'parse.py'] Result:"},
                                                {"type": "text", "text": "def f():\n    return 1\n"}]}], None, None),
}
EFFORTS = [None, "none", "minimal", "low", "medium", "high", "xhigh", "max", 0.0, 0.5, 0.99]


@pytest.fixture(scope="module")
def hf():
    need_model_dir()
    return transformers.AutoTokenizer.from_pretrained(str(MODEL_DIR))


def need_p28k(name):
    if name == "cline_28k" and P28K is None:
        pytest.skip("tools/prompts/p_28k.txt not found in INKLING_S_MODEL_DIR")


def hf_render(hf, msgs, tools, effort, tokenize):
    kw = {} if effort is None else {"reasoning_effort": effort}
    out = hf.apply_chat_template(msgs, tools=tools, tokenize=tokenize, **kw)
    if tokenize and hasattr(out, "keys"):
        out = out["input_ids"]
    return list(out) if tokenize else out


def check(r, hf, vocab, msgs, tools, effort):
    msgs = normalize_messages(msgs)
    got = r.render(msgs, tools, effort)
    want_text = hf_render(hf, msgs, tools, effort, False)
    assert got.text == want_text
    assert got.ids == hf_render(hf, msgs, tools, effort, True)
    assert got.ids == vocab.encode(want_text)                      # plain `tokenizers` on the whole string
    n_asst = sum(m["role"] == "assistant" for m in msgs)
    assert got.spans is not None and len(got.spans) == n_asst
    mask = hf.apply_chat_template(msgs, tools=tools, tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
                                  **({} if effort is None else {"reasoning_effort": effort}))["assistant_masks"]
    ours = [0] * len(got.ids)
    for a, b in got.spans:
        assert got.ids[b - 1] == V.END_SAMPLING
        ours[a:b] = [1] * (b - a)
    assert ours == list(mask)
    return got


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("mode", ["text", "special"])
def test_parity_default_effort(name, mode, hf, renderer, renderer_special, vocab):
    need_p28k(name)
    msgs, tools, effort = CASES[name]
    check(renderer if mode == "text" else renderer_special, hf, vocab, msgs, tools, effort)


@pytest.mark.parametrize("effort", EFFORTS)
@pytest.mark.parametrize("name", ["multi_turn", "tools_full_cycle", "only_system"])
def test_parity_efforts(name, effort, hf, renderer, vocab):
    msgs, tools, _ = CASES[name]
    got = check(renderer, hf, vocab, msgs, tools, effort)
    expect = {None: "0.9", "none": "0", 0.0: "0", 0.5: "0.5"}.get(effort)
    if expect:
        assert f"Thinking effort level: {expect}<|end_message|>" in got.text


def test_28k_prompt_size(renderer):
    need_p28k("cline_28k")
    got = renderer.render(normalize_messages(CASES["cline_28k"][0]), None, None)
    assert 25000 < len(got.ids) < 32768


def test_literal_special_text_in_content(hf, renderer, renderer_special, vocab):
    """A file that contains template markers: same text either way; 'text' mode keeps the markers as text."""
    body = "template line: <|message_model|><|content_text|>hi<|end_message|> and <|content_model_end_sampling|>"
    msgs = normalize_messages([{"role": "user", "content": "read this"},
                               {"role": "assistant", "content": "ok", "tool_calls": [
                                   {"id": "1", "type": "function",
                                    "function": {"name": "run", "arguments": {"cmd": "cat <|end_message|>"}}}]},
                               {"role": "tool", "tool_call_id": "1", "content": body}])
    special = check(renderer_special, hf, vocab, msgs, TOOLS, None)    # transformers / llama.cpp behaviour
    text = renderer.render(msgs, TOOLS, None)
    assert text.text == special.text
    assert text.literals == 2
    assert vocab.decode(text.ids) == special.text
    assert text.ids.count(V.END_MESSAGE) == special.ids.count(V.END_MESSAGE) - 2
    assert text.ids.count(V.END_SAMPLING) == 1                       # only the template's own
    assert text.spans is not None and text.ids[text.spans[0][1] - 1] == V.END_SAMPLING


def test_normalize_messages():
    msgs = normalize_messages([
        {"role": "developer", "content": "d"},
        {"role": "user", "content": ["plain", {"type": "image_url", "image_url": {"url": "data:x"}},
                                     {"type": "input_text", "text": "t"}]},
        {"role": "assistant", "content": None, "reasoning": "r",
         "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": "not json"}},
                        {"id": "2", "function": {"name": "g", "arguments": "[1]"}},
                        {"id": "3", "function": {"name": "h", "arguments": ""}}]}])
    assert msgs[0]["role"] == "system"
    assert [p["text"] for p in msgs[1]["content"]][0] == "plain" and "image omitted" in msgs[1]["content"][1]["text"]
    assert msgs[2]["reasoning_content"] == "r"
    assert [tc["function"]["arguments"] for tc in msgs[2]["tool_calls"]] == [{"raw": "not json"}, {"value": [1]}, {}]
    for bad in ([], [{"role": "bot", "content": "x"}], [{"role": "user", "content": 5}],
                [{"role": "user", "content": [{"type": "audio"}]}]):
        with pytest.raises(BadRequest):
            normalize_messages(bad)


def test_resolve_effort():
    assert resolve_effort({}) is None
    assert resolve_effort({}, "low") == "low"
    assert resolve_effort({"reasoning_effort": "HIGH"}) == "high"
    assert resolve_effort({"reasoning": {"effort": "minimal"}}) == "minimal"
    assert resolve_effort({"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "high"}) == "none"
    assert resolve_effort({"chat_template_kwargs": {"reasoning_effort": "0.3"}}) == 0.3
    for bad in ("huge", 1.5, -1):
        with pytest.raises(BadRequest):
            resolve_effort({"reasoning_effort": bad})
