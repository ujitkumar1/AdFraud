import json
import os
import re
from time import sleep
from typing import List, Dict

import google.generativeai as genai
import pandas as pd
import requests

from src.google_search.config.config import Config
from src.google_search.utils.promt_util import Prompts
from src.google_search.utils.query_validator import QueryValidator
from src.logger import log


class SearchService:
    def __init__(self, gemini_api_key: str, google_search_api_key: str, search_engine_id: str, base_topic: str):
        self.google_search_api_key = google_search_api_key
        self.search_engine_id = search_engine_id
        self.base_topic = base_topic

        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')

        self.validator = QueryValidator()
        self.config = Config()

    def generate_search_queries(self, num_queries_per_technique: int = 10) -> List[str]:
        """
            Generate a list of search queries using zero-shot prompting techniques and apply basic filtering for quality.

            :arg
                num_queries_per_technique : int, optional
                    The number of queries to generate per technique (default is 10).
                    This value is internally doubled for query generation.

            :return
                list
                    A list of generated search queries, filtered for minimum word count and format quality.
                    If generation fails, returns a default set of example queries based on `self.base_topic`.
        """
        num_queries_per_technique *= 2

        def clean_query(query: str) -> str:
            """Clean and format a query string."""
            query = re.sub(r'^[-\d\s•.]*', '', query.strip())
            query = re.sub(r'^[A-Za-z]+:\s*', '', query)
            return query.strip().strip('"')

        def generate_and_filter_queries(prompt: str) -> List[str]:
            try:
                response = self.model.generate_content(prompt)
                queries = [clean_query(q) for q in response.text.strip().split('\n')]

                quality_queries = [q for q in queries if len(q) > 5 and len(q.split()) >= 3]
                return quality_queries
            except Exception as e:
                log.error(f"Error generating queries: {e}")
                return []

        log.info("Generating queries using the Zero Shot technique...")

        prompt = Prompts().zero_shot_prompt(self.base_topic, num_queries_per_technique)
        queries = generate_and_filter_queries(prompt)

        if not queries:
            queries = [
                f'"{self.base_topic} technical implementation"',
                f'"{self.base_topic} methodology"',
                f'"{self.base_topic} case study and analysis"'
            ]

        results = queries

        return results

    def search_google(self, query: str, start_index: int = 1) -> Dict | None:
        """
            Perform a Google search using the Google Custom Search API.

            :arg
                query : str
                    The search query to execute.
                start_index : int, optional
                    The index of the first result to retrieve (default is 1).

            :return
                dict or None
                    Returns a JSON dictionary of search results if the request is successful.
                    Returns None if an error occurs during the request.
        """
        base_url = Config().GOOGLE_URL
        params = {
            'key': self.google_search_api_key,
            'cx': self.search_engine_id,
            'lr': 'lang_en',
            'q': query,
            'start': start_index,
            'siteSearchFilter': 'e'
        }

        try:
            response = requests.get(base_url, params=params)
            return response.json()
        except requests.exceptions.RequestException as e:
            log.error(f"Error executing search: {e}")
            return None

    def collect_resources(self, max_results_per_query: int = 91):
        """
            Collect resources from Google Search results using validated search queries.

            :arg
                max_results_per_query : int, optional
                    The maximum number of search results to collect per query (default is 91).

            :return
                list
                    A list of dictionaries, each containing information about a resource:
                    - 'query': The search query used.
                    - 'title': The title of the search result.
                    - 'url': The URL link to the result.
                    - 'snippet': A brief description or snippet from the result.
                    - 'source': Source of the data ('Google Search API').
        """
        all_resources = []
        queries = self.generate_search_queries()

        validated_queries = self.validator.validate_query_set(queries, self.config.REQUIRED_KEYWORDS)

        if validated_queries['valid']:
            log.info("Proceeding with validated queries:", validated_queries['valid'])

            queries = validated_queries['valid']

            log.info(f"Collecting resources for {len(queries)} queries...")

            for query in queries:
                if not query.strip():
                    continue

                log.info(f"Searching for: {query}")
                start_index = 1

                while start_index <= max_results_per_query:
                    results = self.search_google(query, start_index)

                    if not results or 'items' not in results:
                        log.info(f"No more results for query: {query}")
                        break

                    for item in results['items']:
                        resource = {
                            'query': query,
                            'title': item.get('title', ''),
                            'url': item.get('link', ''),
                            'snippet': item.get('snippet', ''),
                            'source': 'Google Search API'
                        }
                        all_resources.append(resource)
                        log.info(f"Found: {resource['title'][:100]}...")

                    start_index += 10
                    sleep(1)

            log.info(f"Total resources collected: {len(all_resources)}")
            return all_resources

    def save_results(self, resources: List[dict], output_format: str = 'csv'):
        """
            Save the collected resources to a file in the specified format.

            :arg
                resources : List[dict]
                    A list of dictionaries, each representing a resource with details like title, URL, and snippet.
                output_format : str, optional
                    The format to save the file in, either 'csv' or 'json' (default is 'csv').

            :return
                None
                    Saves the resources to a file in the 'data' directory, named either 'research_resources.json' or 'research_resources.csv' based on the specified format.
        """
        os.makedirs('data', exist_ok=True)

        if output_format.lower() == 'json':
            output_path = 'data/research_resources.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(resources, f, indent=4, ensure_ascii=False)
            log.info(f"Results saved to {output_path}")

        elif output_format.lower() == 'csv':
            output_path = 'data/research_resources.csv'
            df = pd.DataFrame(resources)
            df.to_csv(output_path, index=False, encoding='utf-8')
            log.info(f"Results saved to {output_path}")
