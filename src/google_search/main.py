import os

from src.google_search.config.config import Config
from src.google_search.services.search_service import SearchService
from src.google_search.services.file_service import FileService
from src.google_search.utils.scraper import ContentScraper
from src.logger import log

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(base_dir, 'data')

input_path = os.path.join(data_dir, 'research_resources.csv')
output_path = os.path.join(data_dir, 'scraped_content.csv')

def run_google_search():
    """
        Main function to execute the Google search process, including configuration validation, resource collection,
        saving results, and scraping data from URLs in the saved resources.
    """
    log.info("Validating configuration...")
    Config.validate_config()

    log.info("Initializing SearchService...")
    search_service = SearchService(
        gemini_api_key=Config.GOOGLE_API_KEY,
        google_search_api_key=Config.GOOGLE_API_KEY,
        search_engine_id=Config.SEARCH_ENGINE_ID,
        base_topic=Config.BASE_TOPIC
    )

    try:
        log.info("Starting resource collection...")
        resources = search_service.collect_resources(max_results_per_query=91)
        log.info(f"Total resources collected: {len(resources)}")

        log.info("Saving results...")
        FileService().save_results(resources, 'csv')
        log.info("Results saved successfully")

        log.info("Starting URL data scraping...")
        scraper = ContentScraper()
        scraper.scrape_data(
            input_path=input_path,
            output_path=output_path
        )
        log.info("Scraping completed successfully")

    except Exception as e:
        log.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    run_google_search()
