# src/agent.py

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any

from .config import OPENAI_API_KEY

# --- optional deps flags ------------------------------------------------------
try:
    # langchain >= 0.2 + langchain-openai
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    # openai >= 1.0.0
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# --- schema (used for structured outputs) ------------------------------------
if LANGCHAIN_AVAILABLE:
    class ResumeSchema(BaseModel):
        name: Optional[str] = Field(None, description="Full name of the candidate")
        contact_information: Optional[str] = Field(
            None, description="Email/phone/location or other contact details"
        )
        skills: List[str] = Field(default_factory=list, description="List of skills")
        education: List[str] = Field(
            default_factory=list, description="Education entries (free-form)"
        )
        work_experience: List[str] = Field(
            default_factory=list, description="Work experience entries (free-form)"
        )
        years_of_experience: Optional[Union[float, str]] = Field(
            None, description="Total relevant years of experience"
        )


# --- abstraction layer --------------------------------------------------------
class LLMBaseAgent(ABC):
    """Abstract base for resume parsing agents."""

    @abstractmethod
    def parse_resume(self, resume_text: str) -> Dict[str, Any]:
        """Return a dictionary with fields:
        - name
        - contact_information
        - skills (list)
        - education (list)
        - work_experience (list)
        - years_of_experience
        """
        raise NotImplementedError


# --- langchain implementation -------------------------------------------------
class LangChainAgent(LLMBaseAgent):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain-openai is not installed.")
        self.model = model
        self.llm = ChatOpenAI(model=model, temperature=0, openai_api_key=api_key)

        # fallback prompt if structured output isn’t available for any reason
        self._prompt = ChatPromptTemplate.from_template(
            """Extract the following information from the resume below and respond ONLY with a valid, minified JSON object using these exact fields:
- name
- contact_information
- skills (as a list)
- education (as a list)
- work_experience (as a list)
- years_of_experience

Resume text:
{resume_text}
"""
        )

    def parse_resume(self, resume_text: str) -> Dict[str, Any]:
        # Preferred path: structured output (Pydantic)
        try:
            # with_structured_output is available on modern LangChain
            llm_struct = self.llm.with_structured_output(ResumeSchema)  # type: ignore[name-defined]
            result = llm_struct.invoke(
                {
                    "input": f"Extract structured resume data.\n\nResume:\n{resume_text}"
                }
            )
            # result is a Pydantic model instance; convert to dict
            return result.model_dump()  # type: ignore[union-attr]
        except Exception:
            # Fallback: prompt + JSON parsing
            chain = self._prompt | self.llm
            resp = chain.invoke({"resume_text": resume_text})
            return self._parse_json_response(resp.content)

    @staticmethod
    def _parse_json_response(content: str) -> Dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "").strip()
        try:
            obj = json.loads(cleaned)
            return _normalize_resume_dict(obj)
        except Exception:
            return {"raw_response": content}


# --- direct openai implementation --------------------------------------------
class OpenAIAgent(LLMBaseAgent):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai is not installed.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def parse_resume(self, resume_text: str) -> Dict[str, Any]:
        system = (
            "You are an expert HR assistant. "
            "Return ONLY a valid, minified JSON object with the requested fields."
        )
        user = f"""Extract the following information from the resume below and format it as a valid, minified JSON object using these exact fields:
- name
- contact_information
- skills (as a list)
- education (as a list)
- work_experience (as a list)
- years_of_experience

Resume text:
{resume_text}
"""

        # Enforce JSON via response_format
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        try:
            obj = json.loads(content)
            return _normalize_resume_dict(obj)
        except Exception:
            return {"raw_response": content}


# --- factory ------------------------------------------------------------------
def get_resume_parser(preferred: str = "langchain") -> LLMBaseAgent:
    """
    Factory to obtain a resume parser.
    preferred: "langchain" or "openai"
    """
    if preferred == "langchain" and LANGCHAIN_AVAILABLE:
        return LangChainAgent(api_key=OPENAI_API_KEY)
    if preferred == "openai" and OPENAI_AVAILABLE:
        return OpenAIAgent(api_key=OPENAI_API_KEY)
    # fallback preference order
    if LANGCHAIN_AVAILABLE:
        return LangChainAgent(api_key=OPENAI_API_KEY)
    if OPENAI_AVAILABLE:
        return OpenAIAgent(api_key=OPENAI_API_KEY)
    raise ImportError("No suitable backend found. Install `langchain-openai` and/or `openai`.")


# --- helpers ------------------------------------------------------------------
def _normalize_resume_dict(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure consistent types and keys in the result.
    - missing lists become []
    - missing strings become None
    """
    def as_list(v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        # if it’s a single string, split on common separators as a gentle fallback
        if isinstance(v, str):
            parts = [p.strip() for p in v.replace("•", ",").split(",")]
            return [p for p in parts if p]
        return []

    out = {
        "name": obj.get("name"),
        "contact_information": obj.get("contact_information"),
        "skills": as_list(obj.get("skills")),
        "education": as_list(obj.get("education")),
        "work_experience": as_list(obj.get("work_experience")),
        "years_of_experience": obj.get("years_of_experience"),
    }
    return out
