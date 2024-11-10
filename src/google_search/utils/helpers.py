import re

class Helpers:
    @staticmethod
    def clean_query(query: str) -> str:
        """Clean and format a query string."""
        query = re.sub(r'^[-\d\s•.]*', '', query.strip())
        query = re.sub(r'^[A-Za-z]+:\s*', '', query)
        return query.strip().strip('"')