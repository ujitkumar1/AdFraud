import json
import os
from typing import List

import pandas as pd

from src.logger import log


class FileService:
    @staticmethod
    def ensure_directory_exists(directory: str):
        """Ensure that the specified directory exists, creating it if necessary."""
        os.makedirs(directory, exist_ok=True)

    @staticmethod
    def save_to_json(data: List[dict], output_path: str):
        """Save data to a JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            log.info(f"Results saved to {output_path}")
        except Exception as e:
            log.error(f"Failed to save JSON file: {e}")

    @staticmethod
    def save_to_csv(data: List[dict], output_path: str):
        """Save data to a CSV file."""
        try:
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False, encoding='utf-8')
            log.info(f"Results saved to {output_path}")
        except Exception as e:
            log.error(f"Failed to save CSV file: {e}")

    @staticmethod
    def save_results(resources: List[dict], output_format: str = 'csv'):
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
        FileService.ensure_directory_exists('data')

        if output_format.lower() == 'json':
            output_path = 'data/research_resources.json'
            FileService.save_to_json(resources, output_path)
        elif output_format.lower() == 'csv':
            output_path = 'data/research_resources.csv'
            FileService.save_to_csv(resources, output_path)
        else:
            log.error(f"Unsupported output format: {output_format}")
