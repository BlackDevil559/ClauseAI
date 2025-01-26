from langchain.prompts import PromptTemplate

def get_general_query_prompt_template() -> PromptTemplate:
    """
    Generates a prompt template for answering general queries related to contracts using an LLM.

    The template guides the model to analyze the provided context chunks of a contract 
    and generate a well-reasoned answer to the user's query.

    Returns:
        PromptTemplate: A template containing the input variables and instructions for query answering.
    """
    return PromptTemplate(
        input_variables=["query", "context_chunks"],
        template="""You are a legal assistant specialized in contracts. Your task is to answer the user's 
        query based on the provided context chunks of the contract. Analyze the context carefully and 
        provide a precise and accurate response to the query. If the answer cannot be determined from the 
        provided context, say so explicitly.

        QUERY:
        {query}

        CONTEXT CHUNKS TO ANALYZE:
        {context_chunks}

        RESPONSE INSTRUCTIONS:
        1. Use only the information provided in the context chunks to answer the query.
        2. If the query cannot be answered from the provided context, respond with: "The answer is not available in the provided context."
        3. Be concise, clear, and formal in your response.
        4. Do not infer information that is not explicitly stated in the context.

        RESPONSE:""",
    )
