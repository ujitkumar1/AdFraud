import hashlib

from tqdm import tqdm


class DocumentUtils:
    @staticmethod
    def deduplicate_doc(doc):
        """
        Remove duplicate documents based on content uniqueness.

        :arg
            doc : List[Document]
                A list of `Document` objects to be deduplicated.

        :return
            list
                A list of unique `Document` objects, with duplicates removed.
        """
        unique_contents = set()
        deduped_docs = []
        for doc in tqdm(doc):
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in unique_contents:
                unique_contents.add(content_hash)
                deduped_docs.append(doc)

        return deduped_docs

    @staticmethod
    def clean_topic_formatting(topic: str) -> str:
        """
        Remove unsupported formatting characters from a topic string.

        :arg
            topic : str
                The topic string to be cleaned of formatting characters, such as '*'.

        :return
            str
                The cleaned topic string with unsupported characters removed.
        """
        return topic.replace('*', '')
