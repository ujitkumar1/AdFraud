from collections import defaultdict
from typing import List, Dict

import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from src.llm_process.utils.document_utils import DocumentUtils
from src.logger import log


class ClusterUtils:
    random_state = 42

    @staticmethod
    def find_optimal_clusters(embeddings: np.ndarray, min_clusters: int, max_clusters: int) -> int:
        """
        Determine the optimal number of clusters for embeddings using silhouette analysis.

        :arg
            embeddings : np.ndarray
                Array of embeddings to be clustered.
            min_clusters : int
                Minimum number of clusters to consider.
            max_clusters : int
                Maximum number of clusters to consider.

        :return
            int
                The optimal number of clusters based on the highest silhouette score.
        """
        best_score = -1
        optimal_clusters = min_clusters

        for n in tqdm(range(min_clusters, max_clusters + 1)):
            kmeans = KMeans(n_clusters=n, random_state=ClusterUtils.random_state)
            clusters = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, clusters)

            if score > best_score:
                best_score = score
                optimal_clusters = n

        return optimal_clusters

    @staticmethod
    def cluster_documents(embeddings, n_clusters: int):
        """
        Cluster documents into a specified number of clusters.

        :arg
            embeddings : np.ndarray
                Array of document embeddings to be clustered.
            n_clusters : int
                The number of clusters to divide the documents into.

        :return
            np.ndarray
                An array of cluster labels indicating the assigned cluster for each document.
        """
        log.info(f"Clustering documents into {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=ClusterUtils.random_state)
        return kmeans.fit_predict(embeddings)

    @staticmethod
    def merge_cluster_topics(cluster_topics: List[str]) -> Dict:
        """
        Merge and organize topics from multiple clusters into a structured dictionary format.

        :arg
            cluster_topics : List[str]
                A list of topics from different clusters, each in YAML string format.

        :return
            dict
                A dictionary with merged topics, where overlapping topics are combined.
                For lists within topics, duplicates are removed, and non-list values are overwritten with the latest entry.
        """
        merged_topics = defaultdict(dict)

        for topic in cluster_topics:
            cleaned_topic = DocumentUtils().clean_topic_formatting(topic)
            try:
                topic_dict = yaml.safe_load(cleaned_topic)

                if isinstance(topic_dict, dict):
                    for main_topic, details in topic_dict.items():
                        if main_topic not in merged_topics or not isinstance(merged_topics[main_topic], dict):
                            merged_topics[main_topic] = {}

                        if not isinstance(details, dict):
                            merged_topics[main_topic] = details
                        else:
                            for key, value in details.items():
                                if isinstance(value, list):
                                    if key not in merged_topics[main_topic]:
                                        merged_topics[main_topic][key] = []
                                    merged_topics[main_topic][key] = list(
                                        set(merged_topics[main_topic][key] + value)
                                    )
                                else:
                                    merged_topics[main_topic][key] = value

                else:
                    merged_topics[cleaned_topic] = topic_dict

            except yaml.YAMLError as e:
                log.error(f"YAML loading error - {e}")
                continue

        return dict(merged_topics)
