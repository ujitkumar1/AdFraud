import time
import yaml
import numpy as np
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
from src.llm_process.config.llm_config import CONFIG
from src.llm_process.service.document_service import DocumentService
from src.llm_process.utils.cluster_utils import ClusterUtils
from src.llm_process.utils.promt_utils import Prompts
from src.logger import log


class MobileAdFraudAnalyzer:
    def __init__(self, google_api_key: str):
        """
        Initialize the MobileAdFraudAnalyzer with configuration and API key.
        """
        self.document_service = DocumentService(google_api_key)
        self.min_clusters = CONFIG.MIN_CLUSTER
        self.max_clusters = CONFIG.MAX_CLUSTERS
        self.prompts = Prompts()
        self.topic_extraction_chain = self._create_topic_extraction_chain()
        self.hierarchy_generation_chain = self._create_hierarchy_generation_chain()

    def _create_topic_extraction_chain(self):
        """
        Create a chain for extracting key topics from content using LLM.
        """
        return (
            PromptTemplate.from_template(self.prompts.TOPIC_EXTRACTION_PROMPT)
            | self.document_service.llm
            | StrOutputParser()
        )

    def _create_hierarchy_generation_chain(self):
        """
        Create a chain for generating the final YAML hierarchy using LLM.
        """
        return (
            PromptTemplate.from_template(self.prompts.HIERARCHY_GENERATION_PROMPT)
            | self.document_service.llm
            | StrOutputParser()
        )

    def process_documents(self, csv_path: str) -> List:
        """
        Load, process, and deduplicate documents from a CSV file.

        :arg
            csv_path : str
                Path to the CSV file.

        :return
            List[Document]
                A list of deduplicated Document objects.
        """
        log.info("Processing documents from CSV...")
        documents = self.document_service.load_and_process_documents(csv_path)
        log.info(f"Processed {len(documents)} documents.")
        return documents

    def create_document_embeddings(self, documents: List) -> np.ndarray:
        """
        Generate embeddings for the provided documents.

        :arg
            documents : List[Document]

        :return
            np.ndarray
                Document embeddings.
        """
        log.info("Creating document embeddings...")
        embeddings = self.document_service.create_embeddings(documents)
        log.info("Embeddings created.")
        return embeddings

    def extract_topics_from_cluster(self, cluster_docs: List) -> str:
        """
        Extract key topics from a cluster of documents using the topic extraction chain.

        :arg
            cluster_docs : List[Document]

        :return
            str
                Extracted topics as a string.
        """
        combined_content = "\n".join([doc.page_content for doc in cluster_docs[:5]])
        return self.topic_extraction_chain.invoke({"content": combined_content})

    def generate_yaml_hierarchy(self, merged_topics: dict) -> str:
        """
        Generate a YAML hierarchy from merged topics.

        :arg
            merged_topics : dict

        :return
            str
                YAML formatted string.
        """
        log.info("Generating YAML hierarchy...")
        yaml_content = self.hierarchy_generation_chain.invoke({
            "content": yaml.dump(merged_topics, sort_keys=False)
        })
        return yaml_content.replace("*", "")


    def validate_with_llm(self, yaml_content: str, max_attempts: int = 4) -> str:
        """
        Validate and correct YAML content using a language model.

        :arg
            yaml_content : str
            max_attempts : int

        :return
            str
                Validated YAML content.
        """
        attempts = 0
        validated_yaml = yaml_content

        while attempts < max_attempts:
            log.info(f"Validation attempt {attempts + 1}...")
            response = self.document_service.llm.invoke({
                "prompt": (
                    f"Please validate and correct the following YAML structure:\n\n{validated_yaml}"
                )
            })

            if "Error" not in response:
                log.info("Validation successful.")
                return response
            else:
                log.error(f"Validation failed at attempt {attempts + 1}. Retrying...")

            validated_yaml = response
            attempts += 1

        log.error("Validation failed after maximum attempts.")
        return validated_yaml

    def process(self, csv_path: str) -> str:
        """
        Complete pipeline for processing documents and generating the YAML structure.

        :arg
            csv_path : str

        :return
            str
                Validated YAML structure.
        """
        log.info("Starting the analysis process...")
        documents = self.process_documents(csv_path)

        log.info("Creating embeddings for analysis...")
        embeddings = self.create_document_embeddings(documents)

        log.info("Finding optimal number of clusters...")
        n_clusters = ClusterUtils.find_optimal_clusters(embeddings, self.min_clusters, self.max_clusters)

        log.info(f"Clustering documents into {n_clusters} clusters...")
        clusters = ClusterUtils.cluster_documents(embeddings, n_clusters)

        log.info("Extracting topics from each cluster...")
        cluster_topics = []
        for i in tqdm(range(n_clusters), desc="Extracting topics"):
            cluster_docs = [doc for doc, label in zip(documents, clusters) if label == i]
            topics = self.extract_topics_from_cluster(cluster_docs)
            cluster_topics.append(topics)

        log.info("Merging cluster topics...")
        merged_topics = ClusterUtils.merge_cluster_topics(cluster_topics)

        log.info("Generating final YAML hierarchy...")
        final_hierarchy = self.generate_yaml_hierarchy(merged_topics)

        log.info("Validating the YAML structure with LLM...")
        validated_hierarchy = self.validate_with_llm(final_hierarchy)

        try:
            yaml_structure = yaml.safe_load(validated_hierarchy)
            return yaml.dump(yaml_structure, sort_keys=False, indent=2, allow_unicode=True)
        except yaml.YAMLError as e:
            log.error(f"YAML formatting error: {str(e)}")
            return validated_hierarchy