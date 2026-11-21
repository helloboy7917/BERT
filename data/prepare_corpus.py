"""Validate sentence-separated local JSONL and prepare corpus files."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def prepare_corpus(input_path, documents_path, corpus_path):
    input_path, documents_path, corpus_path = map(
        lambda value: Path(value).resolve(), (input_path, documents_path, corpus_path)
    )
    if len({input_path, documents_path, corpus_path}) != 3:
        raise ValueError("Input, documents output, and corpus output must be different files")

    # Validate before opening either output, so invalid input cannot truncate them.
    documents = []
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(document, dict) or not isinstance(document.get("sentences"), list):
                raise ValueError(f"{input_path}:{line_number}: expected a sentences array")
            if any(not isinstance(sentence, str) for sentence in document["sentences"]):
                raise ValueError(f"{input_path}:{line_number}: sentences must contain strings")
            sentences = [sentence.strip() for sentence in document["sentences"] if sentence.strip()]
            if sentences:
                documents.append({"sentences": sentences})
    if not documents:
        raise ValueError(f"{input_path}: no non-empty documents")

    for output in (documents_path, corpus_path):
        output.parent.mkdir(parents=True, exist_ok=True)
    with documents_path.open("w", encoding="utf-8", newline="\n") as docs, \
            corpus_path.open("w", encoding="utf-8", newline="\n") as corpus:
        for document in documents:
            docs.write(json.dumps(document, ensure_ascii=False) + "\n")
            for sentence in document["sentences"]:
                corpus.write(" ".join(sentence.splitlines()) + "\n")
    return len(documents)


def main():
    parser = argparse.ArgumentParser(description="Prepare a local sentence-separated JSONL corpus")
    parser.add_argument("input", help="Input JSONL; one document with a sentences array per line")
    parser.add_argument("--documents", default=str(ROOT / "data" / "documents.jsonl"))
    parser.add_argument("--corpus", default=str(ROOT / "data" / "corpus.txt"))
    args = parser.parse_args()
    count = prepare_corpus(args.input, args.documents, args.corpus)
    print(f"Prepared {count} documents")


if __name__ == "__main__":
    main()
