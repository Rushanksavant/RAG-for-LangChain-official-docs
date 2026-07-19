from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file = ".env", extra = "ignore")

    QDRANT_CLOUD_CLUSTER_ENDPOINT: SecretStr 
    QDRANT_CLOUD_KEY: SecretStr
    HF_API_KEY: SecretStr
    
    GEMINI_API_KEY: SecretStr 
    MCP_SECRET_KEY: SecretStr 
    SERVER_URL: str

    GROQ_API_KEY: SecretStr

    LANGSMITH_API_KEY: SecretStr
    
    max_input_chars: int = 1000  # User query should be at max 250 tokens
    
    ENV: str


settings = Settings()
# print(settings.GEMINI_API_KEY)