class CONFIG:
    LLM_MODEL: str = "gemini-pro"
    TEMPERATURE: float = 0.2

    EMBEDDING_MODEL: str = "models/embedding-001"

    CHUNK_SIZE: int = 3000
    CHUNK_OVERLAP: int = 200

    MIN_CLUSTER: int = 5
    MAX_CLUSTERS: int = 20
