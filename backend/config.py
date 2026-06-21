from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = ""
    deepgram_api_key: str = ""
    redis_url: str = "redis://localhost:6379"
    cors_origins: str = "http://localhost:5173"


settings = Settings()
