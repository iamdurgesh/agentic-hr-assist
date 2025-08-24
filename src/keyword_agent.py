# src/keywords_main.py
from __future__ import annotations

import os
import sys
import json
import argparse
from dotenv import load_dotenv

from keyword_agent import read_document, clean_text, get_keyword_agent


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Keyword/keyphrase extraction from documents using AI agents"
    )
    parser.add_argument("path", help="Path to PDF/DOCX/TXT/MD document")
    parser.add_argument(
        "--backend",
        choices=["langchain", "openai", "auto"],
        default="auto",
        help="Backend to use (auto tries LangChain → OpenAI → local TF-IDF)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of keywords/keyphrases to extract (default: 20)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        print(f"Error: file not found: {args.path}")
        return 1

    print(f"Reading document: {args.path}")
    text = clean_text(read_document(args.path))

    preferred = "langchain" if args.backend == "auto" else args.backend
    agent = get_keyword_agent(preferred=preferred)

    print(f"Extracting top-{args.top_k} keywords/keyphrases with backend: {preferred}")
    result = agent.extract(text, top_k=args.top_k)

    print("\n=== Extracted Keywords/Keyphrases ===\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
