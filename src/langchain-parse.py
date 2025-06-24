from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from .config import OPENAI_API_KEY
import json

def parse_resume_with_gpt(resume_text: str) -> dict:
    """
    Uses GPT-4o via LangChain to extract structured info from resume text.
    Returns a dictionary with the extracted fields.
    """
    # Prompt template for extracting structured resume information
    prompt = ChatPromptTemplate.from_template("""
    Extract the following information from this resume text as a valid JSON object with these fields:
      - name
      - contact_information
      - skills
      - education
      - work_experience
      - years_of_experience

    Resume text:
    {resume_text}
    """)

    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    # Run prompt + LLM
    chain = prompt | llm
    response = chain.invoke({"resume_text": resume_text})

    # Try to parse the response as JSON
    try:
        result = json.loads(response.content)
    except Exception:
        # Fallback: print as raw text if not valid JSON
        result = {"raw_response": response.content}
    return result
