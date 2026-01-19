# genAI_master_start
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Agent Template"
    openai_api_key: str = ""
    model_name: str = "gpt-4o"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
# genAI_master_end
