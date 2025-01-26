from qdrant_client import QdrantClient
class QdrantHandler:
    """
    A class to manage interactions with the Qdrant client.
    """
    def __init__(self, url: str, api_key: str):
        """
        Initialize the QdrantHandler with connection details.
        """
        self.qdrant_client = QdrantClient(url=url, api_key=api_key, timeout=300)

    def load_qdrant_connection(self):
        """
        Get the initialized Qdrant client.
        """
        return self.qdrant_client

    def ensure_collection_exists(self, collection_name: str, vector_size: int):
        """
        Ensure that the specified Qdrant collection exists.
        If it does not exist, create it.
        """
        existing_collections = [
            col.name for col in self.qdrant_client.get_collections().collections
        ]
        if collection_name not in existing_collections:
            print(f"Collection '{collection_name}' does not exist. Creating it...")
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={"distance": "Cosine", "size": vector_size},
            )
            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Collection '{collection_name}' already exists.")

    def store_in_qdrant(self, docs_vector_store, split_documents, collection_name: str):
        """
        Upload the vector embeddings to the specified Qdrant collection.
        """
        self.ensure_collection_exists(collection_name, docs_vector_store.index.d)
        vectors = [
            docs_vector_store.index.reconstruct(i)
            for i in range(docs_vector_store.index.ntotal)
        ]
        payloads = [{"text": doc.page_content} for doc in split_documents]
        self.qdrant_client.upload_collection(
            collection_name=collection_name, vectors=vectors, payload=payloads
        )
        print(f"Data successfully uploaded to Qdrant collection: {collection_name}")

    def search_qdrant(self, collection_name: str, query_vector: list, limit: int = 10, score_threshold: float = 0.5):
        """
        Search for similar vectors in the specified Qdrant collection.

        Args:
            collection_name (str): The name of the Qdrant collection to search in.
            query_vector (list): The query vector to search for.
            limit (int): The maximum number of results to retrieve.
            score_threshold (float): The minimum similarity score threshold for results.

        Returns:
            list: A list of search results with their payloads and similarity scores.
        """
        try:
            results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
            )
            formatted_results = [
                {
                    "id": result.id,
                    "payload": result.payload,
                    "score": result.score,
                }
                for result in results
            ]
            return formatted_results
        except Exception as e:
            print(f"Error during search in Qdrant: {e}")
            return []

    def get_collection_names(self):
        """
        Retrieve and return a list of all collection names in Qdrant.
        
        Returns:
            list: A list of collection names.
        """
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            return collection_names
        except Exception as e:
            print(f"Error fetching collection names: {e}")
            return []
        
    def get_all_payloads(self, collection_name: str):
        """
        Retrieve all payloads and their associated vectors from a Qdrant collection.
        Merge the payloads into a single string.

        Args:
            collection_name (str): The name of the collection to fetch data from.

        Returns:
            str: A string of all merged payloads in the collection.
        """
        merged_payloads = []
        try:
            next_offset = None
            while True:
                # Fetch a batch of records
                scroll_response = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=True,
                )

                # Handle tuple response (for older Qdrant client versions)
                if isinstance(scroll_response, tuple):
                    scroll_result, _ = scroll_response
                    points = scroll_result.points
                    next_offset = scroll_result.next_page_offset
                else:
                    points = scroll_response.points
                    next_offset = scroll_response.next_page_offset

                if not points:
                    break

                for point in points:
                    text = point.payload.get("text", "") if point.payload else ""
                    vector = point.vector
                    merged_payloads.append(f"Vector: {vector}\nText: {text}")

                if not next_offset:
                    break

            return "\n\n".join(merged_payloads)
        except Exception as e:
            print(f"Error fetching payloads and vectors from Qdrant: {e}")
            return ""
