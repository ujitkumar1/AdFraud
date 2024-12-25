from typing import List

from src.google_search.utils.query_utils import QueryUtils
from src.logger import log


class QueriesService:
    @staticmethod
    def generate_and_filter_queries(model, prompt: str) -> List[str]:
        try:
            response = model.generate_content(prompt)
            queries = [QueryUtils().clean_query(q) for q in response.text.strip().split('\n')]

            quality_queries = [q for q in queries if len(q) > 5 and len(q.split()) >= 3]
            return quality_queries
        except Exception as e:
            log.error(f"Error generating queries: {e}")
            return []
