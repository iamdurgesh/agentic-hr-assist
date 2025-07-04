import sys
import os
from dotenv import load_dotenv
from resume_parser import extract_text_from_pdf
from agent import get_resume_parser

def main():
    load_dotenv()

    # Argument parsing
    if len(sys.argv) < 2:
        print("Usage: python main.py path_to_resume.pdf [backend: langchain|openai]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    preferred_backend = sys.argv[2] if len(sys.argv) >= 3 else "langchain"

    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    # Extract resume text
    print(f"\nExtracting text from '{pdf_path}'...")
    resume_text = extract_text_from_pdf(pdf_path)

    # Choose and initialize agent
    print(f"Using backend: {preferred_backend}")
    try:
        parser = get_resume_parser(preferred=preferred_backend)
    except ImportError as e:
        print(f"Agent initialization error: {e}")
        sys.exit(1)

    # Parse resume
    print("\nParsing resume with AI agent...\n")
    result = parser.parse_resume(resume_text)

    print("=== Extracted Candidate Information ===\n")
    print(result)

    # Optional: save to CSV
    if isinstance(result, dict) and "raw_response" not in result and len(sys.argv) >= 4:
        import pandas as pd
        csv_path = sys.argv[3]
        pd.DataFrame([result]).to_csv(csv_path, index=False)
        print(f"\nStructured data saved to {csv_path}")

if __name__ == "__main__":
    main()
