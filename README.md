# ClauseAI

ClauseAI is a document processing and querying application that leverages AI-powered tools for extracting metadata, vectorizing content, and providing intelligent query responses. It combines the power of OpenAI models and Qdrant to create a seamless document management system.

---

## Installation

Follow these steps to set up and run the ClauseAI project locally:

### Prerequisites
- Python 3.10 or higher
- Virtual environment manager (optional but recommended)

### Steps
1. **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd ClauseAI
    ```

2. **Set up a Virtual Environment**:
    ```bash
    python -m venv virtual
    source virtual/bin/activate   # On Windows, use virtual\Scripts\activate
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set Environment Variables**:
    Create a `.env` file in the root directory with the following content:
    ```plaintext
    QDRANT_URL=<your_qdrant_url>
    QDRANT_API_KEY=<your_qdrant_api_key>
    OPENAI_API_KEY=<your_openai_api_key>
    ```

5. **Run the Application**:
    ```bash
    streamlit run workflow.py
    ```

---

## Workflow

ClauseAI consists of two main functionalities:

### 1. **Document Processing**
   - Upload a PDF document.
   - Convert the document to Markdown format and extract metadata.
   - Generate vector embeddings for the document content and store them in Qdrant.
   - Extract entities using GPT-4 for metadata enrichment.

### 2. **Query Document**
   - Select a processed document by its ID.
   - Query the document using two mechanisms:
     - **Qdrant**: Fetch the most relevant context chunks.
     - **LLM**: Refine the Qdrant output using OpenAI GPT-4 for a natural-language response.

---

## Utilities

- **PDF to Markdown Conversion**: Extracts textual content and metadata from uploaded PDF documents.
- **Vectorization**: Converts document content into vector embeddings using OpenAI embeddings and stores them in Qdrant for efficient querying.
- **Entity Extraction**: Uses GPT-4 to identify and extract key entities in the document.
- **Intelligent Querying**: Combines Qdrant's vector search and GPT-4's natural language understanding to deliver detailed query responses.

---

## Contributing

We welcome contributions to ClauseAI! Please fork the repository and create a pull request with your changes.