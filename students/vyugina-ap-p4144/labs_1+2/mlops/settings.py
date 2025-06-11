from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str = "qdrant_api_key"
    qdrant_collection_name: str = "airbnb_listings"
    qdrant_batch_size: int = 64
    qdrant_dimension: int = 47

    data_dir: str = "assets"  # Directory where new data files will be placed


settings = Settings()
