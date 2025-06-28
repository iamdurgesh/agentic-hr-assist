from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from .config import OPENAI_API_KEY
import json

def parse_resume_with_gpt(resume_text: str) -> dict:
    """
    Parses a resume's text using GPT-4o via LangChain to extract structured information.
    Returns a dictionary with fields:
      - name
      - contact_information
      - skills (list)
      - education (list)
      - work_experience (list)
      - years_of_experience (number or string)
    Falls back to raw output if JSON cannot be parsed.
    """
    # Create prompt with strict JSON requirements
    prompt = ChatPromptTemplate.from_template("""
Extract the following information from the resume below and format it as a valid, minified JSON object using these exact fields:
  - name
  - contact_information
  - skills (as a list)
  - education (as a list)
  - work_experience (as a list)
  - years_of_experience

Resume text:
{resume_text}
""")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    # Compose and invoke the agent
    chain = prompt | llm
    response = chain.invoke({"resume_text": resume_text})

    # Try parsing response as JSON, fallback to raw string
    try:
        # Remove any markdown code block formatting if present
        cleaned = response.content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned)
    except Exception:
        result = {"raw_response": response.content}
    return result
