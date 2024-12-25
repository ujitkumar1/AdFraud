import pandas as pd
import numpy as np
from typing import List
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from src.llm_process.utils.document_utils import DocumentUtils
from src.llm_process.config.llm_config import CONFIG
from src.logger import log

class DocumentService:
    def __init__(self, google_api_key: str):
        """
        Initialize the DocumentService with text splitter and embeddings.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG.CHUNK_SIZE,
            chunk_overlap=CONFIG.CHUNK_OVERLAP,
            add_start_index=True
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=CONFIG.EMBEDDING_MODEL,
            google_api_key=google_api_key
        )

        self.llm = GoogleGenerativeAI(
            model=CONFIG.LLM_MODEL,
            google_api_key=google_api_key,
            temperature=CONFIG.TEMPERATURE
        )

    def load_and_process_documents(self, csv_path: str) -> List[Document]:
        """
        Load documents from a CSV file, consolidate content, split, and deduplicate.

        :arg
            csv_path : str
                Path to the CSV file containing document data, with columns 'page_content' and 'page_title'.

        :return
            List[Document]
                A list of deduplicated `Document` objects.
        """
        log.info(f"Loading documents from: {csv_path}")
        df = pd.read_csv(csv_path)
        df['combined_content'] = df['page_content'].fillna('') + ' ' + df['page_title'].fillna('')

        loader = DataFrameLoader(df, page_content_column="combined_content")
        documents = loader.load()

        log.info(f"Loaded {len(documents)} documents. Splitting documents...")
        split_docs = self.text_splitter.split_documents(documents)

        log.info("Deduplicating documents...")
        deduplicated_docs = DocumentUtils.deduplicate_doc(split_docs)

        log.info(f"Processed {len(deduplicated_docs)} unique documents.")
        return deduplicated_docs

    def create_embeddings(self, documents: List[Document]) -> np.ndarray:
        """
        Generate embeddings for a list of documents using Google Generative AI.

        :arg
            documents : List[Document]
                A list of `Document` objects to generate embeddings for.

        :return
            np.ndarray
                An array of embeddings for the documents.
        """
        log.info("Creating embeddings for documents...")
        texts = [doc.page_content for doc in documents]
        embeddings = []

        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            log.info(f"Generating embeddings for batch {i // batch_size + 1}...")
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)

        log.info("Embeddings creation completed.")
        return np.array(embeddings)
