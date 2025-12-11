import os
import requests
from io import BytesIO
from dotenv import load_dotenv
from openai import AzureOpenAI
from PIL import Image


class ImageHelper:
    """Helper class to generate embeddings for images using Azure OpenAI Vision."""

    def __init__(self):
        load_dotenv()
        
        self.api_key = os.getenv("AZURE_OPENAI_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.instance_name = os.getenv("AZURE_OPENAI_INSTANCE_NAME")
        self.vision_deployment = os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT_NAME", "gpt-4-vision")
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT_NAME")
        
        # Construct the endpoint
        self.endpoint = f"https://{self.instance_name}.openai.azure.com/"
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )

    def download_image(self, image_url: str) -> Image.Image:
        """
        Download an image from a URL and return as PIL Image.
        
        Args:
            image_url: URL of the image to download
            
        Returns:
            PIL Image object
        """
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            print(f"Error downloading image from {image_url}: {str(e)}")
            return None

    def analyze_image(self, image_url: str) -> str:
        """
        Analyze an image using Azure OpenAI Vision and return a description.
        
        Args:
            image_url: URL of the image to analyze
            
        Returns:
            Text description of the image content
        """
        try:
            response = self.client.chat.completions.create(
                model=self.vision_deployment,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            },
                            {
                                "type": "text",
                                "text": "Describe what you see in this image in 1-2 sentences, focusing on the main product, its features, and appearance."
                            }
                        ]
                    }
                ],
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error analyzing image from {image_url}: {str(e)}")
            return ""

    def get_image_embedding(self, image_url: str, product_description: str = "") -> list:
        """
        Generate an embedding for an image by analyzing it and embedding the analysis.
        
        Args:
            image_url: URL of the image
            product_description: Optional product description to include in analysis
            
        Returns:
            A list representing the embedding vector, or None if analysis fails
        """
        # Analyze the image to get a text description
        image_analysis = self.analyze_image(image_url)
        
        if not image_analysis:
            return None
        
        # Combine image analysis with product description for richer embedding
        combined_text = f"{image_analysis}. {product_description}" if product_description else image_analysis
        
        # Generate embedding from the combined text
        try:
            response = self.client.embeddings.create(
                input=combined_text,
                model=self.embedding_deployment
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding for image {image_url}: {str(e)}")
            return None

    def get_image_embeddings_batch(self, products: list) -> list:
        """
        Generate embeddings for multiple product images in batch.
        
        Args:
            products: List of dicts with 'image_url' and 'description' keys
            
        Returns:
            List of embedding vectors (same length as input, None for failed items)
        """
        embeddings = []
        
        for product in products:
            image_url = product.get("image_url")
            description = product.get("description", "")
            
            if image_url:
                embedding = self.get_image_embedding(image_url, description)
                embeddings.append(embedding)
            else:
                embeddings.append(None)
        
        return embeddings

    def get_text_embedding_from_image(self, image_url: str) -> str:
        """
        Get rich text description of an image for text-based embedding.
        Useful for hybrid search combining text and image modalities.
        
        Args:
            image_url: URL of the image
            
        Returns:
            Text description of the image
        """
        return self.analyze_image(image_url)


# Singleton instance
_image_helper = None


def get_image_helper():
    """Get or create the ImageHelper singleton instance."""
    global _image_helper
    if _image_helper is None:
        _image_helper = ImageHelper()
    return _image_helper
