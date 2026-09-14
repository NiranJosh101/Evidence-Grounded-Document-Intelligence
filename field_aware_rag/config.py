from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50
    ENCODING_NAME: str = "cl100k_base"
    
    # Phase 2 Settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"  # Fast, accurate 384-dim model
    TOP_K_PER_QUERY: int = 2                       # Chunks per query variant
    TOP_K_PER_FIELD: int = 4
    PINECONE_API_KEY: str = "your-pinecone-api-key"  # API key for Pinecone
    PINECONE_INDEX_NAME: str = "your-pinecone-index-name"  # Name of the Pinecone index

    class Config:
        env_file = ".env"

settings = Settings()