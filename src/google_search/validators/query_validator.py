from typing import List, Dict

from src.logger import log

class QueryValidator:
    @staticmethod
    def validate_query_set(queries: List[str], required_keywords: List[str], min_words: int = 3) -> Dict:
        """
        Validates a list of query strings based on minimum word count and required keywords.

        :arg
            queries : List[str]
                A list of query strings to be validated.
            required_keywords : List[str]
                A list of keywords that each valid query must contain at least one of.
            min_words : int, optional
                Minimum number of words required in each query to be considered valid (default is 3).

        :return
            dict
                A dictionary with two keys:
                - 'valid': List of queries that meet the word count and contain required keywords.
                - 'invalid': List of queries that are either too short, lack required keywords, or missing keywords if not covered.
        """
        keywords_found = {keyword: False for keyword in required_keywords}
        valid_queries = []
        invalid_queries = []

        for query in queries:
            if len(query.split()) < min_words:
                invalid_queries.append(query)
                continue

            query_valid = False
            for keyword in required_keywords:
                if keyword.lower() in query.lower():
                    keywords_found[keyword] = True
                    query_valid = True

            if query_valid:
                valid_queries.append(query)
            else:
                invalid_queries.append(query)

        missing_keywords = [kw for kw, found in keywords_found.items() if not found]
        if missing_keywords:
            log.info(f"Missing keyword coverage for: {', '.join(missing_keywords)}")
            invalid_queries.extend(missing_keywords)

        return {
            'valid': valid_queries,
            'invalid': invalid_queries
        }