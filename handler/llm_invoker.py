import openai
from langchain.prompts import PromptTemplate

from handler.prompt_general_query import get_general_query_prompt_template as get_general_template
from handler.prompt_entity_extractor import get_prompt_template as get_entities_template
from handler.prompt_entity_extractor import get_entities

class GPT4Assistant:
    """
    A class to interact with OpenAI's GPT-4 API.
    """

    def __init__(self, api_key: str):
        """
        Initialize the GPT4Assistant with the OpenAI API key.
        Args:
            api_key (str): The OpenAI API key for authentication.
        """
        self.api_key = api_key
        openai.api_key = self.api_key

    def get_prompt_template(self, task_type: str) -> PromptTemplate:
        """
        Load the appropriate prompt template for the given task type.

        Args:
            task_type (str): The type of task, e.g., "entity_extraction" or "general_query".

        Returns:
            PromptTemplate: The template for the specified task.
        """
        if task_type == "entity_extraction":
            return get_entities_template()
        elif task_type == "general_query":
            return get_general_template()
        else:
            raise ValueError("Invalid task type. Supported types: 'entity_extraction', 'general_query'.")

    def get_response(self, task_type: str, context_chunks: str, query: str) -> str:
        """
        Generate a response from GPT-4 for the specified task using the relevant prompt template.

        Args:
            task_type (str): The type of task, e.g., "entity_extraction" or "general_query".
            context_chunks (str): The context chunks to provide to the prompt.
            kwargs: Additional arguments for the prompt (e.g., entities or query).

        Returns:
            str: The assistant's response.
        """
        try:
            prompt_template = self.get_prompt_template(task_type)

            if task_type == "entity_extraction":
                entities = get_entities()
                if not entities:
                    raise ValueError("For 'entity_extraction', 'entities' must be provided.")
                prompt = prompt_template.format(entities=entities, context_chunks=context_chunks)
            elif task_type == "general_query":
                if not query:
                    raise ValueError("For 'general_query', 'query' must be provided.")
                prompt = prompt_template.format(query=query, context_chunks=context_chunks)
            else:
                raise ValueError("Invalid task type.")

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in contracts."},
                    {"role": "user", "content": prompt},
                ],
            )

            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"An error occurred: {str(e)}"

