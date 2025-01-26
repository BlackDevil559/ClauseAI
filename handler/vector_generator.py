from qdrant_client import QdrantClient
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings


class QdrantDocumentProcessor:
    """
    A class to process Markdown files, generate vector embeddings, and store them in Qdrant.
    """

    def __init__(self, openai_api_key: str, qdrant_client, document_content: str, document_id: str):
        """
        Initialize the QdrantDocumentProcessor class.

        Args:
            openai_api_key (str): OpenAI API key for generating embeddings.
            qdrant_client : Connection to the Qdrant instance.
            mongo_uri (str): MongoDB URI for connecting to the database.
            db_name (str): Name of the database in MongoDB.
            document_id (str): ID of the document to process.
        """
        self.openai_api_key = openai_api_key
        self.qdrant_client = qdrant_client
        self.document_content = document_content
        self.document_id = document_id

    def split_file(self, file_content: str):
        """
        Split the file content into smaller chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_documents = text_splitter.create_documents([file_content])
        return split_documents

    def generate_vector_embeddings(self, split_documents):
        """
        Generate vector embeddings for the split documents using OpenAI embeddings.
        """
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        docs_vector_store = FAISS.from_documents(split_documents, embeddings)
        return docs_vector_store

    def process_document(self):
        """
        Main method to process the document: load, split, embed, and store.
        """
        print(f"Processing document: {self.document_id}")
        try:
            file_content = self.document_content
            split_documents = self.split_file(file_content)
            docs_vector_store = self.generate_vector_embeddings(split_documents)
            self.qdrant_client.store_in_qdrant(docs_vector_store, split_documents, collection_name=self.document_id)
            print(f"Document processing completed successfully for: {self.document_id}")
        except Exception as e:
            raise Exception(f"Error processing document: {e}")
