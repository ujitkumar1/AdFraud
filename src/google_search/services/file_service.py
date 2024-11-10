import json
import os
from typing import List

import pandas as pd

from src.logger import log


class FileService:
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
        os.makedirs('data', exist_ok=True)

        if output_format.lower() == 'json':
            FileService._save_json(resources)
        elif output_format.lower() == 'csv':
            FileService._save_csv(resources)
        else:
            log.error(f"Unsupported output format: {output_format}")

    @staticmethod
    def _save_json(resources: list[dict]) -> None:
        """Helper method to save resources in JSON format."""
        output_path = 'data/research_resources.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(resources, f, indent=4, ensure_ascii=False)
        log.info(f"Results saved to {output_path}")

    @staticmethod
    def _save_csv(resources: list[dict]) -> None:
        """Helper method to save resources in CSV format."""
        output_path = 'data/research_resources.csv'
        df = pd.DataFrame(resources)
        df.to_csv(output_path, index=False, encoding='utf-8')
        log.info(f"Results saved to {output_path}")
