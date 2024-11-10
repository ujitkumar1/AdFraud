import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import PyPDF2
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.logger import log


class ContentScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
        }

    def scrape_data(self, input_path: str = 'data/research_resources.csv',
                    output_path: str = 'data/scraped_content.csv') -> None:
        """
            Scrapes content from URLs in an input CSV file and saves it to an output CSV file.

            :arg
                input_path : str
                    Path to the CSV file with URLs to scrape (default is 'data/research_resources.csv').
                output_path : str
                    Path where scraped data will be saved (default is 'data/scraped_content.csv').

            :return
                None
                Saves the scraped data (page title, URL, and content) into the specified output CSV file.
        """
        resources_df = pd.read_csv(input_path)

        unique_urls = resources_df['url'].unique()
        urls_to_scrape = [url for url in unique_urls if 'youtube.com' not in url.lower()]

        with open(output_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file,
                                    fieldnames=['page_title', 'link', 'page_content'],
                                    escapechar='\\',
                                    quoting=csv.QUOTE_ALL
                                    )
            writer.writeheader()

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(self._scrape_url, url): url for url in urls_to_scrape}

                for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping content"):
                    url = futures[future]
                    try:
                        result = future.result()
                        if result:
                            writer = csv.DictWriter(file,
                                                    fieldnames=['page_title', 'link', 'page_content'],
                                                    escapechar='\\',
                                                    quoting=csv.QUOTE_ALL
                                                    )
                            writer.writerow(result)

                    except Exception as e:
                        log.error(f"Failed to scrape {url}: {e}")

    def _scrape_url(self, url: str):
        """
            Scrape content from a single URL, processing HTML or PDF formats as needed.

            :arg
                url : str
                    The URL to scrape, which can be an HTML page or a PDF file.

            :return
                dict or None
                    Returns a dictionary with the following keys if content is successfully scraped:
                    - 'page_title': Title of the page or document.
                    - 'link': The URL that was scraped.
                    - 'page_content': The main text content from the URL.
                    Returns None if scraping fails.
        """
        try:
            if url.lower().endswith('.pdf'):
                page_title, page_content = self._scrape_pdf(url)
            else:
                page_title, page_content = self._scrape_html(url)

            if page_content:
                return {'page_title': page_title, 'link': url, 'page_content': page_content}
        except Exception as e:
            log.error(f"Error scraping {url}: {e}")
        return None

    def _scrape_html(self, url: str):
        """
            Scrape HTML content from a single URL, extracting the page title and main text content.

            :arg
                url : str
                    The URL of the HTML page to scrape.

            :return
                tuple
                    Returns a tuple with two elements:
                    - page_title : str - The title of the page (or 'No Title' if not found).
                    - page_content : str - The main text content of the page.
                    Returns (None, None) if the request fails or the status code is not 200.
        """
        response = requests.get(url, headers=self.headers, timeout=100)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            page_title = soup.title.string if soup.title else 'No Title'
            page_content = soup.get_text(separator=' ', strip=True)
            return page_title, page_content

        return None, None

    def _scrape_pdf(self, url: str):
        """
            Scrape text content from a PDF file located at a given URL.

            :arg
                url : str
                    The URL of the PDF file to scrape.

            :return
                tuple
                    Returns a tuple with two elements:
                    - page_title : str - The title of the document (set as 'PDF Document').
                    - page_content : str - The extracted text content from all pages of the PDF.
                    Returns (None, None) if the request fails or the status code is not 200.
        """
        response = requests.get(url, headers=self.headers, timeout=100)

        if response.status_code == 200:
            pdf_text = []
            with BytesIO(response.content) as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    pdf_text.append(page.extract_text())

            page_title = 'PDF Document'
            page_content = ' '.join(pdf_text)

            return page_title, page_content

        return None, None
