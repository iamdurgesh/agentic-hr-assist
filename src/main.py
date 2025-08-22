# src/main.py
from __future__ import annotations

import os
import sys
import json
import argparse

from dotenv import load_dotenv

# local modules
from resume_parser import extract_text_from_pdf
from agent import get_resume_parser
from keyword_agent import read_document, clean_text, get_keyword_agent


def cmd_parse_resume(args: argparse.Namespace) -> int:
    """
    Parse a single resume PDF and print structured JSON.
    Optionally save to CSV.
    """
    if not os.path.isfile(args.path):
        print(f"Error: file not found: {args.path}")
        return 1

    print(f"Reading PDF: {args.path}")
    resume_text = extract_text_from_pdf(args.path)

    print(f"Using backend: {args.backend}")
    parser = get_resume_parser(preferred=args.backend)

    print("Parsing resume with AI agent...")
    result = parser.parse_resume(resume_text)

    # pretty-print JSON
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result)

    # optional CSV export
    if args.output and isinstance(result, dict) and "raw_response" not in result:
        try:
            import pandas as pd
            pd.DataFrame([result]).to_csv(args.output, index=False)
            print(f"\nSaved structured data to: {args.output}")
        except Exception as e:
            print(f"CSV save error: {e}")

    return 0


def cmd_extract_keywords(args: argparse.Namespace) -> int:
    """
    Extract keywords/keyphrases from a document with preferred backend.
    """
    if not os.path.isfile(args.path):
        print(f"Error: file not found: {args.path}")
        return 1

    print(f"Reading document: {args.path}")
    doc_text = clean_text(read_document(args.path))

    backend = args.backend
    if backend == "auto":
        # let factory pick best available
        preferred = "langchain"
    else:
        preferred = backend

    print(f"Using backend: {preferred}")
    agent = get_keyword_agent(preferred=preferred)

    print(f"Extracting top-{args.top_k} keywords/keyphrases...")
    result = agent.extract(doc_text, top_k=args.top_k)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AI Agent CLI • Resume parsing and keyword extraction"
    )
    sub = p.add_subparsers(dest="command", required=True)

    # parse-resume
    pr = sub.add_parser(
        "parse-resume",
        help="Parse a resume PDF into structured JSON (optional CSV export).",
    )
    pr.add_argument("path", help="Path to resume PDF")
    pr.add_argument(
        "--backend",
        choices=["langchain", "openai"],
        default="langchain",
        help="LLM backend to use (default: langchain)",
    )
    pr.add_argument(
        "--output",
        help="Optional CSV path to save structured fields",
        default=None,
    )
    pr.set_defaults(func=cmd_parse_resume)

    # extract-keywords
    kw = sub.add_parser(
        "extract-keywords",
        help="Extract keywords/keyphrases from PDF/DOCX/TXT/MD.",
    )
    kw.add_argument("path", help="Path to document")
    kw.add_argument(
        "--backend",
        choices=["langchain", "openai", "auto"],
        default="auto",
        help="Keyword backend (auto tries LangChain→OpenAI→local TF-IDF)",
    )
    kw.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Max keywords/keyphrases to return (default: 20)",
    )
    kw.set_defaults(func=cmd_extract_keywords)

    return p


def main() -> int:
    load_dotenv()  # loads OPENAI_API_KEY from .env if present
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
