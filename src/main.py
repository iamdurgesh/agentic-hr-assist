import sys
from resume_parser import extract_text_from_pdf
from agent import parse_resume_with_gpt

def main(pdf_path):
    print(f"Parsing {pdf_path} ...")
    text = extract_text_from_pdf(pdf_path)
    print("Extracting structured info using AI agent...")
    structured = parse_resume_with_gpt(text)
    print("\n=== Extracted Information ===\n")
    print(structured)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py path_to_resume.pdf")
        sys.exit(1)
    main(sys.argv[1])