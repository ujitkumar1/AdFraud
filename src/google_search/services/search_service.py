from time import sleep
from typing import List, Dict

import google.generativeai as genai
import requests

from src.google_search.config.config import Config
from src.google_search.prompts.prompt_generator import PromptGenerator
from src.google_search.services.file_service import FileService
from src.google_search.services.queries_service import QueriesService
from src.google_search.utils.query_utils import QueryUtils
from src.google_search.validators.query_validator import QueryValidator
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

        log.info("Generating queries using the Zero Shot technique...")

        prompt = PromptGenerator().zero_shot_prompt(self.base_topic, num_queries_per_technique)
        queries = QueriesService.generate_and_filter_queries(self.model, prompt)

        if not queries:
            queries = QueryUtils.default_queries(self.base_topic)

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
            CURRENT_TRIES = 0
            while CURRENT_TRIES <= Config.MAX_TRIES:
                response = requests.get(base_url, params=params)

                if response.status_code == 200:
                    break

                CURRENT_TRIES += 1
                sleep(1)

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
        """Delegate file saving to the FileService."""
        FileService.save_results(resources, output_format)
