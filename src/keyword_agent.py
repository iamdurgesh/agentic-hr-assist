from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Dict, Any

from dotenv import load_dotenv

# local modules
from keyword_agent import read_document, clean_text, get_keyword_agent


def _save_output(result: Dict[str, Any], out_json: str | None, out_csv: str | None) -> None:
    if out_json:
        try:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Saved JSON to: {out_json}")
        except Exception as e:
            print(f"Error saving JSON: {e}")

    if out_csv and isinstance(result, dict) and "raw_response" not in result:
        try:
            import pandas as pd
            # flatten simple lists for CSV friendliness
            row = {
                "keywords": ", ".join(result.get("keywords", []) or []),
                "keyphrases": ", ".join(result.get("keyphrases", []) or []),
            }
            # keep scored lists if present (store as JSON string in a cell)
            if "keywords_scored" in result:
                row["keywords_scored"] = json.dumps(result["keywords_scored"], ensure_ascii=False)
            if "keyphrases_scored" in result:
                row["keyphrases_scored"] = json.dumps(result["keyphrases_scored"], ensure_ascii=False)

            pd.DataFrame([row]).to_csv(out_csv, index=False)
            print(f"Saved CSV to: {out_csv}")
        except Exception as e:
            print(f"Error saving CSV: {e}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract keywords/keyphrases from documents (PDF/DOCX/TXT/MD) using AI agents."
    )
    p.add_argument("path", help="Path to document (PDF/DOCX/TXT/MD)")
    p.add_argument(
        "--backend",
        choices=["langchain", "openai", "auto"],
        default="auto",
        help="Backend to use (auto tries LangChain → OpenAI → local TF-IDF).",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Max number of keywords/keyphrases to return (default: 20).",
    )
    p.add_argument(
        "--out_json",
        default=None,
        help="Optional path to save the full extraction result as JSON.",
    )
    p.add_argument(
        "--out_csv",
        default=None,
        help="Optional path to save a CSV row (keywords/keyphrases flattened).",
    )
    return p


def main() -> int:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        print(f"Error: file not found: {args.path}")
        return 1

    try:
        print(f"Reading document: {args.path}")
        text = clean_text(read_document(args.path))
    except Exception as e:
        print(f"Document read error: {e}")
        return 1

    preferred = "langchain" if args.backend == "auto" else args.backend
    print(f"Using backend: {preferred}")

    try:
        agent = get_keyword_agent(preferred=preferred)
    except Exception as e:
        print(f"Agent initialization error: {e}")
        return 1

    try:
        print(f"Extracting top-{args.top_k} keywords/keyphrases...")
        result = agent.extract(text, top_k=args.top_k)
    except Exception as e:
        print(f"Extraction error: {e}")
        return 1

    # pretty print to console
    if isinstance(result, dict):
        print("\n=== Extracted Keywords/Keyphrases ===\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result)

    # optional outputs
    _save_output(result, args.out_json, args.out_csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
