import sys
import os
import pandas as pd

from dotenv import load_dotenv
from resume_parser import extract_text_from_pdf
from agent import parse_resume_with_gpt

def save_to_csv(data, out_path):
    """
    Saves extracted resume data (dict or JSON string) to a CSV file.
    """
    if isinstance(data, str):
        import json
        data = json.loads(data)
    df = pd.DataFrame([data])
    df.to_csv(out_path, index=False)
    print(f"\nData saved to {out_path}")

def main():
    load_dotenv()
    if len(sys.argv) < 2:
        print("Usage: python main.py path_to_resume.pdf [output.csv]")
        sys.exit(1)
    pdf_path = sys.argv[1]
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print(f"Parsing {pdf_path} ...")
    try:
        resume_text = extract_text_from_pdf(pdf_path)
        print("Extracting structured info using AI agent...")
        structured = parse_resume_with_gpt(resume_text)
        print("\n=== Extracted Information ===\n")
        print(structured)
        # Optional: Save to CSV if output path is provided
        if len(sys.argv) >= 3:
            out_csv = sys.argv[2]
            save_to_csv(structured, out_csv)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()