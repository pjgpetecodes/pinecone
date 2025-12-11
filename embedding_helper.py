import os
from dotenv import load_dotenv
from openai import AzureOpenAI


class EmbeddingHelper:
    """Helper class to generate embeddings using Azure OpenAI."""

    def __init__(self):
        load_dotenv()
        
        self.api_key = os.getenv("AZURE_OPENAI_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.instance_name = os.getenv("AZURE_OPENAI_INSTANCE_NAME")
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT_NAME")
        
        # Construct the endpoint
        self.endpoint = f"https://{self.instance_name}.openai.azure.com/"
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )

    def get_embedding(self, text: str) -> list:
        """
        Generate an embedding for the given text using Azure OpenAI.
        
        Args:
            text: The text to embed
            
        Returns:
            A list representing the embedding vector
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_deployment
        )
        return response.data[0].embedding

    def get_embeddings_batch(self, texts: list) -> list:
        """
        Generate embeddings for a batch of texts using Azure OpenAI.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            A list of embedding vectors
        """
        response = self.client.embeddings.create(
            input=texts,
            model=self.embedding_deployment
        )
        
        # Sort by index to maintain order
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [e.embedding for e in embeddings]
