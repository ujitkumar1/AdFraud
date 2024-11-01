import os

from dotenv import load_dotenv

from src.llm_process.service.fradu_analyzer import MobileAdFraudAnalyzer
from src.logger import log


def run_llm_process():
    """
    Execute the main LLM-based processing workflow, including loading configurations,
    processing documents, and saving the final YAML hierarchy.

    :arg
        None

    :return
        None
    """

    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    analyzer = MobileAdFraudAnalyzer(api_key)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')

    input_file = os.path.join(data_dir, 'scraped_content.csv')
    output_dir = os.path.join(data_dir, 'output')
    output_file = os.path.join(output_dir, "mobile_ad_fraud_hierarchy.yaml")

    yaml_hierarchy = analyzer.process(input_file)

    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding='utf-8') as f:
        f.write(yaml_hierarchy)

    log.info(f"Generated YAML hierarchy saved to {output_file}")


if __name__ == "__main__":
    run_llm_process()
