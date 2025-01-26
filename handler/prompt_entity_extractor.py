from langchain.prompts import PromptTemplate

def get_entities() -> list[str]:
    """
    Fetches the list of predefined entities for extraction based on the document type.

    Args:
        documentType (str): Classified Document Type for which entities to be extracted.

    Returns:
        list[str]: A list of entity names.
    """
    predefined_entities = {
        "general_contract": [
            "Contract Title",
            "Effective Date",
            "Parties Involved",
            "Term of Contract",
            "Termination Clause",
            "Confidentiality Clause",
            "Governing Law",
            "Payment Terms",
            "Signatories",
        ]
    }
    return predefined_entities.get("general_contract", [])

def get_prompt_template() -> PromptTemplate:
    """
    Generates a prompt template for extracting entities from a classified document using an LLM.

    The template guides the model to identify and extract specific entities from the document
    based on the classified document type and provided context chunks.

    Returns:
        PromptTemplate: A template containing the input variables and extraction instructions.
    """
    return PromptTemplate(
        input_variables=["entities", "context_chunks"],
        template="""You are an advanced entity extractor. Your task is to extract specific entities 
        from the context chunks of the document. These chunks are representative of the document and 
        have been retrieved using cosine similarity.

        DOCUMENT TYPE: General Contract

        ENTITIES TO EXTRACT:
        {entities}

        CONTEXT CHUNKS TO ANALYZE:
        {context_chunks}

        Please provide your extracted entities in the following JSON format:
        {{
            "extracted_entities": {{
                "entity_name": "extracted_value",
                "entity_name_2": "extracted_value_2",
                ...
            }}
        }}

        Remember:
        1. Extract only the specified entities relevant to the document type.
        2. If an entity is not found, set its value as "None".
        3. The extracted values must be directly based on the provided context chunks.

        Extraction:""",
    )
