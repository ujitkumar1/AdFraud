import os
from typing import List

from dotenv import load_dotenv

load_dotenv()


class Config:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    SEARCH_ENGINE_ID = os.getenv('GOOGLE_SEARCH_ID')

    GOOGLE_URL: str = "https://www.googleapis.com/customsearch/v1"

    BASE_TOPIC: str = "Ad fraud in mobile marketing"
    REQUIRED_KEYWORDS: List[str] = ["ad fraud", "prevention", "ad networks", "legal", "economic impact"]

    MAX_TRIES: int = 4

    @staticmethod
    def validate_config():
        if not Config.GOOGLE_API_KEY or not Config.SEARCH_ENGINE_ID:
            raise ValueError("Required environment variables not found. Please check .env file.")
