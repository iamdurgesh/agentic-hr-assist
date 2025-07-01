import os
import json
from .config import OPENAI_API_KEY

# Optionally use LangChain if desired
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Fallback: direct OpenAI API
import openai

def parse_resume_with_gpt(resume_text: str, use_langchain: bool = True) -> dict:
    """
    Parse resume text using GPT-4o via either LangChain or OpenAI API directly.
    Returns a dictionary of structured candidate info.
    Falls back to raw output if not valid JSON.
    """
    prompt_text = """
Extract the following information from the resume below and format it as a valid, minified JSON object using these exact fields:
  - name
  - contact_information
  - skills (as a list)
  - education (as a list)
  - work_experience (as a list)
  - years_of_experience

Resume text:
{resume_text}
"""

    if use_langchain and LANGCHAIN_AVAILABLE:
        # Use LangChain pipeline
        prompt = ChatPromptTemplate.from_template(prompt_text)
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        chain = prompt | llm
        response = chain.invoke({"resume_text": resume_text})
        content = response.content
    else:
        # Direct OpenAI API fallback
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        messages = [
            {"role": "system", "content": "You are an expert HR assistant. Only respond with valid, minified JSON as requested."},
            {"role": "user", "content": prompt_text.format(resume_text=resume_text)}
        ]
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0
        )
        content = completion.choices[0].message.content.strip()

    # Try parsing as JSON (strip markdown code blocks if present)
    try:
        cleaned = content
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "").strip()
        result = json.loads(cleaned)
    except Exception:
        result = {"raw_response": content}
    return result
