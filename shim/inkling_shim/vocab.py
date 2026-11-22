"""Inkling-2-Mini's special tokens and a small wrapper around the `tokenizers` tokenizer.

Two encoders share one vocabulary: `encode()` treats special-token text as special (what transformers and llama.cpp do
with a rendered prompt), `encode_text()` treats it as plain text (used for anything a client wrote, see prompt.py).
`token_bytes()` gives the raw bytes of a byte-level BPE token, so the output parser can decode UTF-8 incrementally.
"""
from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer

ENDOFTEXT = 199999
MSG_USER = 200000
MSG_MODEL = 200001
MSG_SYSTEM = 200002
MSG_TOOL = 200003
CONTENT_TEXT = 200004
CONTENT_IMAGE = 200005
END_SAMPLING = 200006          # eos: the model's end of turn
CONTENT_THINKING = 200008
END_MESSAGE = 200010
CONTENT_TOOL_ERROR = 200022
CONTENT_XML = 200024
CONTENT_TOOL_JSON = 200049
CONTENT_TOOL_TEXT = 200057

STOP_IDS = (END_SAMPLING, ENDOFTEXT)   # the engine stops on exactly these two (ink_serve.h is_stop_tok)


def _byte_level_table() -> dict[str, int]:
    """GPT-2's bytes_to_unicode, inverted: the printable stand-in character -> the byte it stands for."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class Vocab:
    def __init__(self, model_dir: str | Path):
        path = str(Path(model_dir) / "tokenizer.json")
        self.tok = Tokenizer.from_file(path)
        self.tok_text = Tokenizer.from_file(path)
        self.tok_text.encode_special_tokens = True      # special-token text in a client's string stays text
        added = json.loads(Path(path).read_text(encoding="utf-8"))["added_tokens"]
        self.special = {a["content"]: a["id"] for a in added if a.get("special")}
        self.special_text = {v: k for k, v in self.special.items()}
        table = _byte_level_table()
        self._bytes: dict[int, bytes] = {}
        for piece, i in self.tok.get_vocab(with_added_tokens=False).items():
            try:
                self._bytes[i] = bytes(table[ch] for ch in piece)
            except KeyError:                              # not a byte-level piece (none expected)
                self._bytes[i] = piece.encode("utf-8")

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_text(self, text: str) -> list[int]:
        return self.tok_text.encode(text, add_special_tokens=False).ids if text else []

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=False)

    def is_special(self, i: int) -> bool:
        return i in self.special_text

    def token_bytes(self, i: int) -> bytes:
        b = self._bytes.get(i)
        if b is None:
            b = self.special_text.get(i, "").encode("utf-8")
        return b
