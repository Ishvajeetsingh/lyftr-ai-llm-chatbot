import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure LiteLLM never makes network calls for cost maps or telemetry at import time
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_TELEMETRY"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"


class Settings(BaseSettings):
    LLM_API_KEYS: str = "key_1,key_2"
    DATABASE_URL: str = "sqlite:///./data/app.db"
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    ENABLE_FAILURE_SIMULATION: bool = True
    LLM_MODEL: str = "gemini/gemini-2.0-flash"
    DEFAULT_SYSTEM_PROMPT: str = "You are a helpful AI assistant."

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def api_keys_list(self) -> List[str]:
        keys = [k.strip() for k in self.LLM_API_KEYS.split(",") if k.strip()]
        return keys if keys else ["key_1"]

    @property
    def allowed_origins_list(self) -> List[str]:
        origins = [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]
        return origins if origins else ["*"]


settings = Settings()
