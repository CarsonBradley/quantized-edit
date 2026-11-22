import json
import os
import sys
from pathlib import Path

import pytest

SHIM = Path(__file__).resolve().parent.parent
# The model's tokenizer.json and chat_template.jinja are not part of this repository. Point INKLING_S_MODEL_DIR at
# the folder that holds them (the pack's Hugging Face folder); without it, every test that needs them is skipped.
MODEL_DIR = Path(os.environ["INKLING_S_MODEL_DIR"]) if os.environ.get("INKLING_S_MODEL_DIR") else None
FAKE_ENGINE = SHIM / "tests" / "fake_engine.py"
sys.path.insert(0, str(SHIM))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from inkling_shim import vocab as V  # noqa: E402
from inkling_shim.prompt import Renderer  # noqa: E402


def need_model_dir():
    if MODEL_DIR is None or not (MODEL_DIR / "tokenizer.json").is_file() or not (MODEL_DIR / "chat_template.jinja").is_file():
        pytest.skip("set INKLING_S_MODEL_DIR to the folder with tokenizer.json and chat_template.jinja")


@pytest.fixture(scope="session")
def vocab():
    need_model_dir()
    return V.Vocab(MODEL_DIR)


@pytest.fixture(scope="session")
def renderer(vocab):
    return Renderer(MODEL_DIR, vocab=vocab, literals="text")


@pytest.fixture(scope="session")
def renderer_special(vocab):
    return Renderer(MODEL_DIR, vocab=vocab, literals="special")


def reply(vocab, *, think=None, text=None, calls=(), end=True):
    """Token ids of a model turn: optional thinking, optional text, tool calls [(name, args_json_text)]."""
    ids = []
    if think is not None:
        ids += [V.MSG_MODEL, V.CONTENT_THINKING] + vocab.encode_text(think) + [V.END_MESSAGE]
    if text is not None:
        ids += [V.MSG_MODEL, V.CONTENT_TEXT] + vocab.encode_text(text) + [V.END_MESSAGE]
    for name, args in calls:
        ids += [V.MSG_MODEL] + vocab.encode_text(name) + [V.CONTENT_TOOL_JSON]
        ids += vocab.encode_text('{"name":' + json.dumps(name) + ',"args":' + args + "}") + [V.END_MESSAGE]
    if end:
        ids.append(V.END_SAMPLING)
    return ids


class FakeProbes:
    """memguard.SystemProbes without the machine: every reading is an attribute the test sets
    (paused= and healthy= are stored as is_paused / is_healthy)."""

    def __init__(self, **kw):
        self.is_paused, self.is_healthy, self.ooms, self.free, self.level = False, True, 7, 80, 1
        self.procs = {}                   # prefix -> set of pids
        self.ollama, self.holder = [], None
        if "paused" in kw:
            self.is_paused = kw.pop("paused")
        if "healthy" in kw:
            self.is_healthy = kw.pop("healthy")
        self.__dict__.update(kw)

    def paused(self):
        return self.is_paused

    def healthy(self):
        return self.is_healthy

    def watch_count(self):
        return self.ooms

    def free_pct(self):
        return self.free

    def pressure_level(self):
        return self.level

    def processes(self, prefix):
        return set(self.procs.get(prefix, ()))

    def ollama_models(self):
        return list(self.ollama)

    def lock_holder(self):
        return self.holder
