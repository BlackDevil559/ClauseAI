import streamlit as st
from handler.layout_identifier import PDFToMarkdownConverter
from handler.qdrant_adapter import QdrantHandler
from handler.vector_generator import QdrantDocumentProcessor
from handler.query_retrieval import QdrantQueryHandler
from handler.llm_invoker import GPT4Assistant
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.sidebar.title("Control Panel")
page = st.sidebar.radio("Choose a page:", ["Process Document", "Query Document"])

qdrant_handler = QdrantHandler(url=QDRANT_URL, api_key=QDRANT_API_KEY)

if page == "Process Document":
    st.title("ClauseAI")

    uploaded_file = st.file_uploader("Upload a Document", type=["pdf"])
    if uploaded_file:
        pdf_path = f"/tmp/{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Document '{uploaded_file.name}' uploaded successfully!")

        st.write("**Step 1: Converting Document to Markdown...**")
        converter = PDFToMarkdownConverter(pdf_path)
        markdown_file, markdown_content = converter.convert()
        st.code(markdown_content[:500], language="markdown")

        st.write("**Step 2: Generating Vector Embeddings...**")
        processor = QdrantDocumentProcessor(
            OPENAI_API_KEY, qdrant_handler, markdown_content, markdown_file
        )
        processor.process_document()
        st.success("Vector embeddings generated successfully!")

        st.write("**Step 3: Extracting Entities...**")
        assistant = GPT4Assistant(OPENAI_API_KEY)
        response = assistant.get_response(
            task_type="entity_extraction", context_chunks=markdown_content, query=""
        )
        st.json(response)
        st.success("Entities extracted successfully!")

if page == "Query Document":
    st.title("ClauseAI")

    st.write("**Step 1: Select a Document ID**")
    document_ids = qdrant_handler.get_collection_names()

    if not document_ids:
        st.warning("No documents found in Qdrant. Please process a document first.")
    else:
        selected_document_id = st.selectbox("Select a Document ID:", document_ids)

        query = st.text_input("Enter your question about the document:")

        if query and selected_document_id:
            st.write("**Step 2: Query Mechanism**")
            query_client = QdrantQueryHandler(
                document_id=selected_document_id,
                openai_api_key=OPENAI_API_KEY,
                qdrant_client=qdrant_handler
            )
            qdrant_response = query_client.query_response(query)

            st.write("**Qdrant Query Response:**")
            st.json(qdrant_response)

            context_chunks = [result["payload"] for result in qdrant_response]

            assistant = GPT4Assistant(OPENAI_API_KEY)
            refined_response = assistant.get_response(
                task_type="general_query",
                context_chunks=context_chunks,
                query=query
            )

            st.write("**LLM Refined Response:**")
            st.write(refined_response)
