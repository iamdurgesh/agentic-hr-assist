
from agent import get_resume_parser

_parser = get_resume_parser(preferred="langchain")

def parse_resume(resume_text: str) -> dict:
    """
    Backward-compatible wrapper.
    Prefer using `agent.get_resume_parser(...).parse_resume(...)` directly.
    """
    return _parser.parse_resume(resume_text)
