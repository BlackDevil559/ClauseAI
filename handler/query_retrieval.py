from typing import List, Dict
from langchain_community.embeddings import OpenAIEmbeddings
class QdrantQueryHandler:
    """
    A class to handle querying and retrieving responses from Qdrant using vector embeddings.
    """
    def __init__(self, document_id: str, openai_api_key: str, qdrant_client):
        """
        Initialize the QdrantQueryHandler with required parameters.

        Args:
            document_id (str): The ID of the document in Qdrant.
            openai_api_key (str): The OpenAI API key for generating vector embeddings.
            qdrant_client: The Qdrant client instance.
        """
        self.document_id = document_id
        self.qdrant_client = qdrant_client
        self.openai_api_key = openai_api_key

    def query_response(self, prompt: str, limit: int = 10, score_threshold: float = 0.1) -> List[Dict]:
        """
        Generate a query response by searching the Qdrant collection.

        Args:
            prompt (str): The user input or query prompt to generate embeddings.
            limit (int): The maximum number of results to retrieve (default is 10).
            score_threshold (float): The minimum similarity score threshold for results.

        Returns:
            List[Dict]: A list of search results with payloads and similarity scores.
        """
        query_vector = self.generate_vector_embedding(prompt)
        
        query_results = self.qdrant_client.search_qdrant(
            collection_name=self.document_id,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold
        )
        return query_results
    
    def generate_vector_embedding(self, text: str):
        """
        Generate a vector embedding for a given string using OpenAI embeddings.

        Args:
            text (str): The text or string for which the embedding needs to be generated.

        Returns:
            List[float]: The generated vector embedding.
        """
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        vector = embeddings.embed_query(text)
        return vector
    