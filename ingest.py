"""Build a document index from your own files, so `ask.py` and `serve.py` run over them
instead of the SQuAD demo corpus.

    python ingest.py path/to/docs --name "Company Handbook"

Reads .txt and .md files, splits them into overlapping chunks, and writes data/index.json.
(PDF/HTML are intentionally out of scope to keep this zero-dependency; convert to text first.)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

CHUNK_WORDS = 180
OVERLAP_WORDS = 40
INDEX = os.path.join(os.path.dirname(__file__), "data", "index.json")


def _chunk(text: str) -> list[str]:
    """Paragraph-aware sliding window over words."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [text] if text else []
    out, step = [], CHUNK_WORDS - OVERLAP_WORDS
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + CHUNK_WORDS])
        if chunk:
            out.append(chunk)
        if i + CHUNK_WORDS >= len(words):
            break
    return out


def build(folder: str, name: str) -> dict:
    files = []
    for ext in ("txt", "md", "markdown"):
        files += glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True)
    if not files:
        raise SystemExit(f"No .txt/.md files found under {folder!r}")
    contexts, sources = [], []
    for path in sorted(files):
        with open(path, encoding="utf-8", errors="replace") as f:
            for chunk in _chunk(f.read()):
                contexts.append(chunk)
                sources.append(os.path.relpath(path, folder))
    return {"name": name, "contexts": contexts, "sources": sources}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder of .txt/.md documents")
    ap.add_argument("--name", default="custom documents", help="label for this corpus")
    args = ap.parse_args()
    index = build(args.folder, args.name)
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    json.dump(index, open(INDEX, "w"))
    print(f"Indexed {len(index['contexts'])} chunks from "
          f"{len(set(index['sources']))} files -> {INDEX}")
    print("Now run:  python serve.py   (or: python ask.py \"your question\")")


if __name__ == "__main__":
    main()
