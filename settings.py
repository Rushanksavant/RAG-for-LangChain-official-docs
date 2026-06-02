from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file = ".env", extra = "ignore")

    GEMINI_API_KEY: SecretStr ## doesn't reveal on print
    ENV: str


settings = Settings()
print(settings.GEMINI_API_KEY)