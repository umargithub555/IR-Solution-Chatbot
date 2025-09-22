from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    api_key: str = Field(..., alias="GROQ_API_KEY")
    hugging_face_token: str = Field(..., alias="HUGGINGFACE_HUB_TOKEN")
    primary_model: str = Field(..., alias="PRIMARY_MODEL")
    fallback_model: str = Field(..., alias="FALLBACK_MODEL")

    knowledge_docs_path: str = "./knowledge_base/"
    processed_data_path: str = "./processed_data"

    combined_processed_data_path: str = "./combined_processed_embeddings"

    separate_processed_data_path: str = "./separate_processed_embeddings"

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra="ignore",   # ignore unknown vars instead of erroring
    )



@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings class.
    This ensures that settings are loaded only once,
    improving performance.
    """
    return Settings()


settings = get_settings()
