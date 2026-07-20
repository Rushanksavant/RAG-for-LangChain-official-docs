from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file = ".env", extra = "ignore")
    
    GEMINI_API_KEY: SecretStr 
    MCP_SECRET_KEY: SecretStr 
    SERVER_URL: str

    GROQ_API_KEY: SecretStr

    LANGCHAIN_API_KEY: SecretStr
    LANGCHAIN_PROJECT: str = "LangGraph-RAG-Agent"
    LANGCHAIN_TRACING_V2: bool
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    
    
    max_input_chars: int = 1000  # User query should be at max 250 tokens



settings = Settings()
# print(settings.GEMINI_API_KEY)