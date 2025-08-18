from __future__ import annotations

import os
import re
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

# Load your key from the shared config
try:
    from .config import OPENAI_API_KEY
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# =========================
# Optional dependencies
# =========================
# Document loaders
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except Exception:
    PYMUPDF_AVAILABLE = False

try:
    import docx  # python-docx
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

# LangChain OpenAI
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# OpenAI (>=1.0)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Local fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# =========================
# Document reading
# =========================
def read_document(path: str) -> str:
    """
    Read PDF / DOCX / TXT / MD into plain text.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF not installed. `pip install pymupdf`")
        return _read_pdf(path)
    elif ext == ".docx":
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not installed. `pip install python-docx`")
        return _read_docx(path)
    elif ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, TXT, or MD.")


def _read_pdf(path: str) -> str:
    chunks = []
    with fitz.open(path) as doc:
        for page in doc:
            chunks.append(page.get_text())
    return "\n".join(chunks)


def _read_docx(path: str) -> str:
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# =========================
# Normalization helpers
# =========================
def _normalize_keywords_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize to:
      - keywords: list[str]
      - keyphrases: list[str]
      (optionally keep *_scored if present)
    """

    def to_list_of_text(v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            out = []
            for it in v:
                if isinstance(it, str):
                    t = it.strip()
                    if t:
                        out.append(t)
                elif isinstance(it, dict):
                    t = str(it.get("text", "")).strip()
                    if t:
                        out.append(t)
            return _dedupe_keep_order(out)
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(",")]
            return _dedupe_keep_order([p for p in parts if p])
        return []

    payload: Dict[str, Any] = {
        "keywords": to_list_of_text(obj.get("keywords")),
        "keyphrases": to_list_of_text(obj.get("keyphrases")),
    }

    # keep scored outputs if provided
    if isinstance(obj.get("keywords_scored"), list):
        payload["keywords_scored"] = obj["keywords_scored"]
    if isinstance(obj.get("keyphrases_scored"), list):
        payload["keyphrases_scored"] = obj["keyphrases_scored"]

    return payload


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            out.append(it)
            seen.add(it)
    return out


# =========================
# Abstraction
# =========================
class KeywordAgentBase(ABC):
    @abstractmethod
    def extract(self, text: str, top_k: int = 20) -> Dict[str, Any]:
        """
        Returns dict with:
          - keywords (list[str])
          - keyphrases (list[str])
          Optionally:
          - keywords_scored
          - keyphrases_scored
        """
        raise NotImplementedError


# =========================
# LangChain implementation
# =========================
if LANGCHAIN_AVAILABLE:
    class KeywordSchema(BaseModel):
        keywords: List[str] = Field(default_factory=list)
        keyphrases: List[str] = Field(default_factory=list)
        keywords_scored: Optional[List[Dict[str, Any]]] = None
        keyphrases_scored: Optional[List[Dict[str, Any]]] = None


class LangChainKeywordAgent(KeywordAgentBase):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain-openai not installed.")
        self.llm = ChatOpenAI(model=model, temperature=0, openai_api_key=api_key)
        self.prompt = ChatPromptTemplate.from_template(
            """Extract salient keywords and multi-word keyphrases from the document below.
Return ONLY a valid, minified JSON with fields:
- keywords: list of short keywords
- keyphrases: list of multi-word phrases
Optionally:
- keywords_scored: list of {{ "text": str, "score": number in [0,1] }}
- keyphrases_scored: list of {{ "text": str, "score": number in [0,1] }}

Limit each list to a maximum of {top_k} unique items. No commentary.

Document:
{doc_text}
"""
        )

    def extract(self, text: str, top_k: int = 20) -> Dict[str, Any]:
        # Try structured output first
        try:
            llm_struct = self.llm.with_structured_output(KeywordSchema)  # type: ignore[name-defined]
            res = llm_struct.invoke(
                {"input": f"Extract top-{top_k} keywords and keyphrases.\n\n{text}"}
            )
            return _normalize_keywords_payload(res.model_dump())  # type: ignore[union-attr]
        except Exception:
            # Fallback: prompt + JSON parse
            chain = self.prompt | self.llm
            resp = chain.invoke({"doc_text": text, "top_k": top_k})
            content = resp.content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            try:
                obj = json.loads(content)
                payload = _normalize_keywords_payload(obj)
                # truncate to top_k if needed
                payload["keywords"] = payload["keywords"][:top_k]
                payload["keyphrases"] = payload["keyphrases"][:top_k]
                return payload
            except Exception:
                return {"raw_response": resp.content}


# =========================
# OpenAI implementation
# =========================
class OpenAIKeywordAgent(KeywordAgentBase):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not installed.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def extract(self, text: str, top_k: int = 20) -> Dict[str, Any]:
        system = "You are an NLP assistant. Return ONLY valid, minified JSON."
        user = f"""Extract salient keywords and keyphrases from the document below.
Return ONLY JSON with:
- keywords: list of short keywords
- keyphrases: list of multi-word phrases
(Optional) scored lists:
- keywords_scored: [{{"text": str, "score": number in [0,1]}}]
- keyphrases_scored: [{{"text": str, "score": number in [0,1]}}]

Limit to {top_k} unique items per list. No commentary.

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
            payload = _normalize_keywords_payload(obj)
            payload["keywords"] = payload["keywords"][:top_k]
            payload["keyphrases"] = payload["keyphrases"][:top_k]
            return payload
        except Exception:
            return {"raw_response": content}


# =========================
# Local TF-IDF fallback
# =========================
class LocalTFIDFKeywordAgent(KeywordAgentBase):
    def extract(self, text: str, top_k: int = 20) -> Dict[str, Any]:
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not installed for TF-IDF fallback.")
        # Basic TF-IDF over a single doc (scores are just term weights)
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
        )
        X = vectorizer.fit_transform([text])
        scores = X.toarray().ravel()
        vocab = vectorizer.get_feature_names_out()

        idxs = np.argsort(scores)[::-1]
        keywords = []
        scored = []
        for i in idxs:
            term = vocab[i]
            if scores[i] <= 0:
                break
            keywords.append(term)
            scored.append({"text": term, "score": float(scores[i])})
            if len(keywords) >= top_k:
                break

        keyphrases = _top_bigrams(text, top_k)
        return {
            "keywords": keywords,
            "keyphrases": keyphrases,
            "keywords_scored": scored,
        }


def _top_bigrams(text: str, top_k: int = 20) -> List[str]:
    # Very simple bigram ranker
    tokens = [w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z\-]+", text)]
    counts: Dict[Tuple[str, str], int] = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [" ".join(bi) for bi, _ in ranked]


# =========================
# Factory
# =========================
def get_keyword_agent(preferred: str = "langchain"):
    """
    Return the best available keyword agent in priority order.
    """
    if preferred == "langchain" and LANGCHAIN_AVAILABLE:
        return LangChainKeywordAgent(api_key=OPENAI_API_KEY)
    if preferred == "openai" and OPENAI_AVAILABLE:
        return OpenAIKeywordAgent(api_key=OPENAI_API_KEY)

    # Fallback order
    if LANGCHAIN_AVAILABLE:
        return LangChainKeywordAgent(api_key=OPENAI_API_KEY)
    if OPENAI_AVAILABLE:
        return OpenAIKeywordAgent(api_key=OPENAI_API_KEY)
    if SKLEARN_AVAILABLE:
        return LocalTFIDFKeywordAgent()

    raise ImportError(
        "No keyword backend available. Install `langchain-openai` or `openai` or `scikit-learn`."
    )


# =========================
# Tiny CLI (optional)
# =========================
def _cli():
    parser = argparse.ArgumentParser(description="Keyword/keyphrase extractor")
    parser.add_argument("path", help="Path to PDF/DOCX/TXT/MD")
    parser.add_argument("--backend", choices=["langchain", "openai", "auto"], default="auto")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    text = clean_text(read_document(args.path))
    preferred = "langchain" if args.backend in ("langchain", "auto") else args.backend
    agent = get_keyword_agent(preferred=preferred)
    result = agent.extract(text, top_k=args.top_k)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
