import os
import requests
from io import BytesIO
from typing import Optional, List
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


class ImageHelper:
    """
    Helper class to generate embeddings for images using CLIP (offline).
    Uses local CLIP model for completely offline image processing.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", target_dim: int = 1536):
        """
        Initialize CLIP model for offline image embeddings.
        
        Args:
            model_name: HuggingFace model identifier for CLIP
                       Default: openai/clip-vit-base-patch32 (produces 512-dim embeddings)
                       Alternative: openai/clip-vit-large-patch14 (produces 768-dim embeddings)
            target_dim: Target dimension for embeddings (default 1536 to match Azure OpenAI)
                       CLIP embeddings will be padded with zeros to match this dimension
        """
        print(f"Loading CLIP model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_dim = self.model.config.projection_dim
        self.target_dim = target_dim
        print(f"CLIP model loaded. Native dimension: {self.clip_dim}, Target dimension: {self.target_dim}")

    def download_image(self, image_url: str) -> Optional[Image.Image]:
        """
        Download an image from a URL and return as PIL Image.
        
        Args:
            image_url: URL of the image to download
            
        Returns:
            PIL Image object or None if download fails
        """
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            print(f"Error downloading image from {image_url}: {str(e)}")
            return None

    def get_image_embedding(self, image_url: str) -> Optional[List[float]]:
        """
        Generate an embedding for an image using CLIP (offline).
        
        Args:
            image_url: URL of the image
            
        Returns:
            List representing the embedding vector (512 or 768 dims), or None if fails
        """
        # Download image
        image = self.download_image(image_url)
        if image is None:
            return None
        
        try:
            # Process image with CLIP
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate image embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # Convert to list
            embedding = image_features[0].cpu().numpy().tolist()
            
            # Pad embedding to target dimension if needed
            if len(embedding) < self.target_dim:
                embedding = embedding + [0.0] * (self.target_dim - len(embedding))
            
            return embedding
        
        except Exception as e:
            print(f"Error generating embedding for image {image_url}: {str(e)}")
            return None

    def get_image_embeddings_batch(self, products: List[dict]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple product images in batch.
        
        Args:
            products: List of dicts with 'image_url' key
            
        Returns:
            List of embedding vectors (same length as input, None for failed items)
        """
        embeddings = []
        
        for i, product in enumerate(products, 1):
            image_url = product.get("image_url")
            
            if image_url:
                print(f"  Processing {i}/{len(products)}: {product.get('title', 'Unknown')}...")
                embedding = self.get_image_embedding(image_url)
                embeddings.append(embedding)
            else:
                embeddings.append(None)
        
        return embeddings

    def get_text_image_embedding(self, text_or_image) -> Optional[List[float]]:
        """
        Generate embeddings for either text or image using CLIP's unified space.
        CLIP can embed both text and images in the same semantic space.
        
        Args:
            text_or_image: Either a text string or image URL
            
        Returns:
            Embedding vector in CLIP space
        """
        # If it looks like a URL, process as image
        if isinstance(text_or_image, str) and text_or_image.startswith(('http://', 'https://')):
            return self.get_image_embedding(text_or_image)
        
        # Otherwise process as text
        try:
            inputs = self.processor(text=text_or_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            embedding = text_features[0].cpu().numpy().tolist()
            
            # Pad embedding to target dimension if needed
            if len(embedding) < self.target_dim:
                embedding = embedding + [0.0] * (self.target_dim - len(embedding))
            
            return embedding
        except Exception as e:
            print(f"Error generating text embedding: {str(e)}")
            return None

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1)
        """
        emb1 = torch.tensor(embedding1, device=self.device)
        emb2 = torch.tensor(embedding2, device=self.device)
        
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)
        return similarity.item()


# Singleton instance
_image_helper = None


def get_image_helper():
    """Get or create the ImageHelper singleton instance."""
    global _image_helper
    if _image_helper is None:
        _image_helper = ImageHelper()
    return _image_helper

