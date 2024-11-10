import re
from typing import List
from src.logger import log

class QueryHelper:
    @staticmethod
    def generate_and_filter_queries(model, prompt: str) -> List[str]:
        """
        Generate queries using the model and filter them for quality.

        :arg
            model : object
                The language model used for query generation.
            prompt : str
                The prompt used to generate queries.

        :return
            list of str
                Filtered list of generated queries.
        """
        try:
            response = model.generate_content(prompt)
            queries = [QueryHelper.clean_query(q) for q in response.text.strip().split('\n')]

            quality_queries = [q for q in queries if len(q) > 5 and len(q.split()) >= 3]
            return quality_queries
        except Exception as e:
            log.error(f"Error generating queries: {e}")
            return []


    @staticmethod
    def clean_query(query: str) -> str:
        """Clean and format a query string."""
        query = re.sub(r'^[-\d\s•.]*', '', query.strip())
        query = re.sub(r'^[A-Za-z]+:\s*', '', query)
        return query.strip().strip('"')

    @staticmethod
    def default_queries(base_topic: str) -> List[str]:
        """
        Return a default set of queries based on the base topic.

        :arg
            base_topic : str
                The base topic for the default queries.

        :return
            list of str
                A list of default example queries.
        """
        return [
            f'"{base_topic} technical implementation"',
            f'"{base_topic} methodology"',
            f'"{base_topic} case study and analysis"'
        ]