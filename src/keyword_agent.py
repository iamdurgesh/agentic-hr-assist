# src/keyword_agent.py
from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from .config import OPENAI_API_KEY

# ---------- optional deps ----------
try:
    import fitz  # PyMuPDF for PDFs
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import docx  # python-docx for .docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ---------- simple local fallback (no LLM) ----------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =======================
# Document reading utils
# =======================

def read_document(path: str) -> str:
    """
    Read PDF, DOCX, or TXT into plain text.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required for PDF reading. Install `pymupdf`.")
        return _read_pdf(path)
    elif ext == ".docx":
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx is required for DOCX reading. Install `python-docx`.")
        return _read_docx(path)
    elif ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF/DOCX/TXT.")


def _read_pdf(path: str) -> str:
    text = []
    with fitz.open(path) as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)


def _read_docx(path: str) -> str:
    document = docx.Document(path)
    return "\n".join(p.text for p in document.paragraphs)


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# =======================
# Keyword schema helpers
# =======================

def _normalize_keywords_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize keywords/keyphrases to:
      - keywords: list[str]
      - keyphrases: list[str]
    Optionally keep scores if provided as list of {text, score}.
    """
    def extract_text_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            out = []
            for item in value:
                if isinstance(item, str):
                    t = item.strip()
                    if t:
                        out.append(t)
                elif isinstance(item, dict):
                    t = str(item.get("text", "")).strip()
                    if t:
                        out.append(t)
            return out
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",")]
            return [p for p in parts if p]
        return []

    payload: Dict[str, Any] = {}
    payload["keywords"] = extract_text_list(obj.get("keywords"))
    payload["keyphrases"] = extract_text_list(obj.get("keyphrases"))
    # keep optional scores if present and well-formed
    if isinstance(obj.get("keywords_scored"), list):
        payload["keywords_scored"] = obj["keywords_scored"]
    if isinstance(obj.get("keyphrases_scored"), list):
        payload["keyphrases_scored"] = obj["keyphrases_scored"]
    return payload


# =======================
# Abstraction (Agent API)
# =======================

class KeywordAgentBase(ABC):
    @abstractmethod
    def extract(self, text: str, top_k: int = 20) -> Dict[str, Any]:
        """
        Return {
          "keywords": [str, ...],
          "keyphrases": [str, ...],
          # optionally:
          "keywords_scored": [{"text": str, "score": float}, ...],
          "keyphrases_scored": [{"text": str, "score": float}, ...]
        }
        """
        raise NotImplementedError


# ---------- LangChain agent ----------
if LANGCHAIN_AVAILABLE:
    class KeywordSchema(BaseModel):
        keywords: List[str] = Field(default_factory=list, description="Unigrams or short keywords")
        keyphrases: List[str] = Field(default_factory=list, description="Multi-word phrases")
        # optional scored outputs
        keywords_scored: Optional[List[Dict[str, Any]]] = Field(default=None)
        keyphrases_scored: Optional[List[Dict[str, Any]]] = Field(default=None)

class LangChainKeywordAgent(KeywordAgentBase):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain-openai not installed.")
        self.llm = ChatOpenAI(model=model, temperature=0, openai_api_key=api_key)
        self.prompt = ChatPromptTemplate.from_template(
            """Extract salient keywords and multi-word keyphrases from the document text below.
Return ONLY a valid, minified JSON with fields:
- keywords: list of single- or short-word keywords
- keyphrases: list of multi-word phrases
Optionally include:
- keywords_scored: list of objects {{ "text": string, "score": number in [0,1] }}
- keyphrases_scored: list of objects {{ "text": string, "score": number in [0,1] }}

Limit to a maximum of {top_k} items per list. Avoid duplicates. No commentary.

Document:
{doc_text}
"""
        )

    def extract(self, text: str, top_k: int = 20) -> Dict[str, Any]:
        # try structured output first
        try:
            llm_struct = self.llm.with_structured_output(KeywordSchema)  # type: ignore[name-defined]
            res = llm_struct.invoke(
                {
                    "input": f"Extract top-{top_k} keywords and keyphrases.\n\nDocument:\n{text}"
                }
            )
            return _normalize_keywords_payload(res.model_dump())  # type: ignore[union-attr]
        except Exception:
            # fallback to prompt + JSON parse
            chain = self.prompt | self.llm
            resp = chain.invoke({"doc_text": text, "top_k": top_k})
            content = resp.content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            try:
                obj = json.loads(content)
                return _normalize_keywords_payload(obj)
            except Exception:
                return {"raw_response": resp.content}


# ---------- Direct OpenAI agent ----------
class OpenAIKeywordAgent(KeywordAgentBase):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not installed.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def extract(self, text: str, top_k: int = 20) -> Dict[str, Any]:
        system = (
            "You are an NLP assistant. Return ONLY valid, minified JSON with the requested fields."
        )
        user = f"""Extract salient keywords and multi-word keyphrases from the document text below.
Return ONLY a valid, minified JSON with fields:
- keywords: list of single- or short-word keywords
- keyphrases: list of multi-word phrases
Optionally include:
- keywords_scored: list of objects {{ "text": string, "score": number in [0,1] }}
- keyphrases_scored: list of objects {{ "text": string, "score": number in [0,1] }}

Limit to a maximum of {top_k} items per list. Avoid duplicates. No commentary.

Document:
{text}
"""
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
            return _normalize_keywords_payload(obj)
        except Exception:
            return {"raw_response": content}


# ---------- Local TF-IDF fallback (no LLM) ----------
class LocalTFIDFKeywordAgent(KeywordAgentBase):
    """
    Extremely simple TF-IDF keyword extractor as a last resort.
    Not SOTA, but gives deterministic output when LLMs are unavailable.
    """

    def extract(self, text: str, top_k: int = 20) -> Dict[str, Any]:
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not installed; cannot run local TF-IDF fallback.")
        import numpy as np

        # very naive token pattern; you may tune this
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
        )
        X = vectorizer.fit_transform([text])
        scores = X.toarray().ravel()
        idxs = np.argsort(scores)[::-1][:top_k]
        vocab = vectorizer.get_feature_names_out()
        keywords = [vocab[i] for i in idxs if scores[i] > 0]

        # heuristic: keyphrases (two-word combos) by scanning text
        phrases = _top_bigrams(text, top_k=top_k)

        return {
            "keywords": keywords,
            "keyphrases": phrases,
            "keywords_scored": [{"text": k, "score": float(scores[idx])} for k, idx in zip(keywords, idxs[: len(keywords)])],
        }


def _top_bigrams(text: str, top_k: int = 20) -> List[str]:
    words = [w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z\-]+", text)]
    counts: Dict[Tuple[str, str], int] = {}
    for i in range(len(words) - 1):
        pair = (words[i], words[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [" ".join(p[0]) for p in ranked]


# ---------- factory ----------
def get_keyword_agent(preferred: str = "langchain"):
    """
    Returns the best available keyword agent.
    """
    if preferred == "langchain" and LANGCHAIN_AVAILABLE:
        return LangChainKeywordAgent(api_key=OPENAI_API_KEY)
    if preferred == "openai" and OPENAI_AVAILABLE:
        return OpenAIKeywordAgent(api_key=OPENAI_API_KEY)

    # fallback order
    if LANGCHAIN_AVAILABLE:
        return LangChainKeywordAgent(api_key=OPENAI_API_KEY)
    if OPENAI_AVAILABLE:
        return OpenAIKeywordAgent(api_key=OPENAI_API_KEY)
    if SKLEARN_AVAILABLE:
        return LocalTFIDFKeywordAgent()

    raise ImportError("No keyword backend available. Install `langchain-openai` or `openai` or `scikit-learn`.")
