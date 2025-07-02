import os
import json
from abc import ABC, abstractmethod
from .config import OPENAI_API_KEY

# Optional imports for LangChain and OpenAI
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ----------- Abstraction Layer -----------

class LLMBaseAgent(ABC):
    """
    Abstract LLM agent interface for resume parsing.
    """

    @abstractmethod
    def parse_resume(self, resume_text: str) -> dict:
        pass


class LangChainAgent(LLMBaseAgent):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.prompt_template = ChatPromptTemplate.from_template("""
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
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=0,
            openai_api_key=self.api_key
        )

    def parse_resume(self, resume_text: str) -> dict:
        chain = self.prompt_template | self.llm
        response = chain.invoke({"resume_text": resume_text})
        content = response.content
        return self._parse_json_response(content)

    def _parse_json_response(self, content: str) -> dict:
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return {"raw_response": content}


class OpenAIAgent(LLMBaseAgent):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model

    def parse_resume(self, resume_text: str) -> dict:
        prompt = """
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
        system_message = {
            "role": "system",
            "content": "You are an expert HR assistant. Only respond with valid, minified JSON as requested."
        }
        user_message = {
            "role": "user",
            "content": prompt.format(resume_text=resume_text)
        }
        client = openai.OpenAI(api_key=self.api_key)
        completion = client.chat.completions.create(
            model=self.model,
            messages=[system_message, user_message],
            temperature=0
        )
        content = completion.choices[0].message.content.strip()
        return self._parse_json_response(content)

    def _parse_json_response(self, content: str) -> dict:
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return {"raw_response": content}


# ----------- Factory Function -----------

def get_resume_parser(preferred: str = "langchain") -> LLMBaseAgent:
    """
    Returns the desired resume parsing agent.
    :param preferred: "langchain" or "openai"
    :return: instance of LLMBaseAgent
    """
    if preferred == "langchain" and LANGCHAIN_AVAILABLE:
        return LangChainAgent(api_key=OPENAI_API_KEY)
    elif preferred == "openai" and OPENAI_AVAILABLE:
        return OpenAIAgent(api_key=OPENAI_API_KEY)
    elif LANGCHAIN_AVAILABLE:
        return LangChainAgent(api_key=OPENAI_API_KEY)
    elif OPENAI_AVAILABLE:
        return OpenAIAgent(api_key=OPENAI_API_KEY)
    else:
        raise ImportError("No available LLM agent backend found. Please install langchain and/or openai.")

