import re


class QueryUtils:
    @staticmethod
    def clean_query(query: str) -> str:
        """Clean and format a query string."""
        query = re.sub(r'^[-\d\s•.]*', '', query.strip())
        query = re.sub(r'^[A-Za-z]+:\s*', '', query)
        return query.strip().strip('"')

    @staticmethod
    def default_queries(base_topic: str) -> list:
        """ Default queries for a base topic."""
        return [
            f'"{base_topic} technical implementation"',
            f'"{base_topic} methodology"',
            f'"{base_topic} case study and analysis"'
        ]
