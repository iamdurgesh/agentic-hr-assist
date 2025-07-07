import sys
import os
from dotenv import load_dotenv
from resume_parser import extract_text_from_pdf
from agent import get_resume_parser

def main():
    load_dotenv()  # Load .env for API keys

    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python main.py <resume.pdf> [backend: langchain|openai] [output.csv]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    preferred_backend = sys.argv[2] if len(sys.argv) >= 3 else "langchain"
    output_csv = sys.argv[3] if len(sys.argv) >= 4 else None

    # Check if PDF exists
    if not os.path.isfile(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    # Extract text from PDF
    print(f"Extracting text from '{pdf_path}'...")
    resume_text = extract_text_from_pdf(pdf_path)

    # Get the agent
    print(f"Using agent backend: {preferred_backend}")
    try:
        parser = get_resume_parser(preferred=preferred_backend)
    except ImportError as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)

    # Parse resume
    print("Parsing resume using AI agent...\n")
    result = parser.parse_resume(resume_text)

    print("=== Extracted Candidate Information ===\n")
    if isinstance(result, dict):
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result)

    # Optionally, save to CSV
    if output_csv and isinstance(result, dict) and "raw_response" not in result:
        try:
            import pandas as pd
            pd.DataFrame([result]).to_csv(output_csv, index=False)
            print(f"\nStructured data saved to '{output_csv}'")
        except Exception as e:
            print(f"Error saving to CSV: {e}")

if __name__ == "__main__":
    main()
