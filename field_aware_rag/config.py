from dotenv import find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50
    ENCODING_NAME: str = "cl100k_base"

    # Embedding & Pinecone Settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "evident-index"
    TOP_K_PER_QUERY: int = 2
    TOP_K_PER_FIELD: int = 4

    # OpenRouter Settings
    OPENROUTER_API_KEY: str = ""
    DEFAULT_MODEL: str = "openrouter/free"

    # Pydantic v2 Settings Configuration
    model_config = SettingsConfigDict(
        env_file=find_dotenv(".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()