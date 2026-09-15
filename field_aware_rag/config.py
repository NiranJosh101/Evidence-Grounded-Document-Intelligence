import os
from pydantic_settings import BaseSettings

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

    # Gemini Settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DEFAULT_MODEL: str = "gemini-2.5-flash"

    class Config:
        env_file = ".env"

settings = Settings()