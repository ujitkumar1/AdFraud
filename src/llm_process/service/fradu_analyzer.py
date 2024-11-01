import time
from typing import List

import numpy as np
import pandas as pd
import yaml
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.llm_process.config.llm_config import CONFIG
from src.llm_process.utils.cluster_utils import ClusterUtils
from src.llm_process.utils.document_utils import DocumentUtils
from src.llm_process.utils.promt_utils import Prompts
from src.logger import log


class MobileAdFraudAnalyzer:
    def __init__(self, google_api_key: str, min_clusters: int = CONFIG.MIN_CLUSTER,
                 max_clusters: int = CONFIG.MAX_CLUSTERS, ):
        """Initialize the analyzer with improved clustering parameters."""
        self.llm = GoogleGenerativeAI(
            model=CONFIG.LLM_MODEL,
            google_api_key=google_api_key,
            temperature=CONFIG.TEMPERATURE
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=CONFIG.EMBEDDING_MODEL,
            google_api_key=google_api_key
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG.CHUNK_SIZE,
            chunk_overlap=CONFIG.CHUNK_OVERLAP,
            add_start_index=True
        )

        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.topic_extraction_chain = self._create_topic_extraction_chain()
        self.hierarchy_generation_chain = self._create_hierarchy_generation_chain()

    def load_and_process_documents(self, csv_path: str) -> List[Document]:
        """
            Load documents from a CSV file, consolidate content, and deduplicate.

            :arg
                csv_path : str
                    Path to the CSV file containing document data, with columns 'page_content' and 'page_title'.

            :return
                List[Document]
                    A list of deduplicated `Document` objects with combined content from the CSV file.
                    Each document combines 'page_content' and 'page_title', then splits and removes duplicates.
        """
        df = pd.read_csv(csv_path)
        df['combined_content'] = df['page_content'].fillna('') + ' ' + df['page_title'].fillna('')

        loader = DataFrameLoader(df, page_content_column="combined_content")
        documents = loader.load()

        split_docs = self.text_splitter.split_documents(documents)

        deduped_docs = DocumentUtils().deduplicate_doc(split_docs)

        return deduped_docs

    def create_document_embeddings(self, documents: List[Document]) -> np.ndarray:
        """
        Generate embeddings for a list of documents.

        :arg
            documents : List[Document]
                A list of `Document` objects, each containing text content to be embedded.

        :return
            np.ndarray
                A NumPy array containing the embeddings for each document.
        """
        log.info("Creating document embeddings...")
        texts = [doc.page_content for doc in documents]
        embeddings = []
        batch_size = 10
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)
            time.sleep(1)
        return np.array(embeddings)

    def _create_topic_extraction_chain(self):
        """Create a chain for extracting key topics from content."""
        return (
                PromptTemplate.from_template(Prompts().TOPIC_EXTRACTION_PROMPT)
                | self.llm
                | StrOutputParser()
        )

    def _create_hierarchy_generation_chain(self):
        """Create a chain for generating the final YAML hierarchy."""
        return (
                PromptTemplate.from_template(Prompts().HIERARCHY_GENERATION_PROMPT)
                | self.llm
                | StrOutputParser()
        )

    def _extract_topics_from_cluster(self, cluster_docs: List[Document]) -> str:
        combined_content = "\n".join([doc.page_content for doc in cluster_docs[:5]])
        return self.topic_extraction_chain.invoke({"content": combined_content})

    def validate_with_llm(self, yaml_content: str, max_attempts: int = 4) -> str:
        """
        Validate and correct YAML content using a language model, attempting up to a maximum number of times.

        :arg
            yaml_content : str
                The YAML content to be validated and corrected.
            max_attempts : int, optional
                The maximum number of validation attempts to make (default is 4).

        :return
            str
                Returns the validated YAML content as a properly formatted string.
                If validation fails after all attempts, returns the last attempted YAML structure.
        """
        attempts = 0
        validated_hierarchy = yaml_content

        while attempts < max_attempts:
            log.info(f"Attempting validation attempt {attempts + 1}...")

            response = self.llm(
                prompt=(
                    f"Please convert the following YAML structure by treating each section and subsection header (currently marked "
                    f"with #) as YAML keys. Ensure that each header is nested appropriately under its parent section, "
                    f"and return a properly formatted YAML structure without any additional comments or explanations.\n\n"
                    f"YAML content:\n{validated_hierarchy}"
                )
            )

            if "Error" not in response:
                log.info("Validation succeeded.")
                return response
            else:
                log.error(f"Validation failed in attempt {attempts + 1}. Applying corrections...")

            validated_hierarchy = response
            attempts += 1

        log.info("Validation attempts exhausted. Using last available YAML structure.")
        return validated_hierarchy

    def process(self, csv_path: str) -> str:
        """
        Process a CSV file of documents through a pipeline to extract and organize topics, outputting a validated YAML structure.

        :arg
            csv_path : str
                Path to the CSV file containing documents to be processed.

        :return
            str
                A formatted YAML string representing the organized topic hierarchy based on extracted topics and clusters.
        """
        documents = self.load_and_process_documents(csv_path)

        log.info("Creating embeddings...")
        embeddings = self.create_document_embeddings(documents)

        log.info("Finding optimal number of clusters...")
        n_clusters = ClusterUtils().find_optimal_clusters(embeddings, self.min_clusters, self.max_clusters)

        log.info(f"Clustering documents into {n_clusters} clusters...")
        clusters = ClusterUtils().cluster_documents(embeddings, n_clusters)

        log.info("Extracting topics from clusters...")
        cluster_topics = []
        for i in tqdm(range(n_clusters)):
            cluster_docs = [doc for doc, label in zip(documents, clusters) if label == i]
            topics = self._extract_topics_from_cluster(cluster_docs)
            cluster_topics.append(topics)

        log.info("Merging and organizing topics...")
        merged_topics = ClusterUtils().merge_cluster_topics(cluster_topics)

        final_hierarchy = self.hierarchy_generation_chain.invoke({
            "content": yaml.dump(merged_topics, sort_keys=False)
        })

        final_hierarchy = final_hierarchy.replace("*", "")

        validated_hierarchy = self.validate_with_llm(final_hierarchy)

        try:
            yaml_structure = yaml.safe_load(validated_hierarchy)
            return yaml.dump(yaml_structure, sort_keys=False, indent=2, allow_unicode=True)
        except yaml.YAMLError as e:
            log.error(f"Error in final YAML generation: {str(e)}")
            return validated_hierarchy
