"""
Face Embedding Helper using DeepFace (Offline)
Generates facial embeddings for comparing user profile images.
Uses local facial recognition - no API calls needed.

EDUCATIONAL PURPOSE: Demonstrates vector embedding distance measurement
for user similarity. Not intended for real-world identification or tracking.
"""

import os
import requests
from io import BytesIO
from typing import Optional, List
from PIL import Image
import numpy as np

try:
    from deepface import DeepFace
except ImportError:
    print("DeepFace not installed. Install with: pip install deepface opencv-python-headless")
    raise


class FaceEmbeddingHelper:
    """
    Generate facial embeddings from profile images using DeepFace (offline).
    Embeddings are 128-dimensional vectors capturing facial characteristics.
    """

    def __init__(self, model_name: str = "Facenet", cache_dir: str = "data/store/user_images"):
        """
        Initialize face embedding model.
        
        Args:
            model_name: DeepFace model to use
                       Options: "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace"
                       Default: "Facenet" (128-dim embeddings)
            cache_dir: Directory to cache downloaded profile images
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.embedding_dim = 128  # Facenet produces 128-dimensional embeddings
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"Initializing DeepFace model: {model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Cache directory: {self.cache_dir}")
        
        # Test model initialization
        try:
            # DeepFace initializes models lazily, we'll handle errors when extracting
            pass
        except Exception as e:
            print(f"Warning: DeepFace initialization: {e}")

    def download_image(self, image_url: str, user_id: str = None) -> Optional[Image.Image]:
        """
        Download an image from a URL and return as PIL Image.
        Caches images locally for reuse.
        
        Args:
            image_url: URL of the image
            user_id: User ID for caching (e.g., "USER-001")
            
        Returns:
            PIL Image object or None if download fails
        """
        # Check cache first
        if user_id:
            cache_filename = f"{user_id}.jpg"
            cache_path = os.path.join(self.cache_dir, cache_filename)
            
            if os.path.exists(cache_path):
                try:
                    image = Image.open(cache_path).convert('RGB')
                    return image
                except Exception as e:
                    print(f"Error loading cached image {cache_path}: {str(e)}")
        
        # Download from URL
        try:
            response = requests.get(image_url, timeout=15)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')

            # Cache the image
            if user_id:
                try:
                    image.save(cache_path, 'JPEG', quality=95)
                except Exception as e:
                    print(f"Warning: Could not cache image to {cache_path}: {str(e)}")

            return image
        except Exception as e:
            # Fallback: retry without query params (Unsplash variants sometimes 404 with params)
            base_url = image_url.split('?')[0]
            if base_url != image_url:
                try:
                    response = requests.get(base_url, timeout=15)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert('RGB')

                    if user_id:
                        try:
                            image.save(cache_path, 'JPEG', quality=95)
                        except Exception as e2:
                            print(f"Warning: Could not cache image to {cache_path}: {str(e2)}")

                    return image
                except Exception as e2:
                    print(f"Error downloading image from {image_url}: {str(e)} | Fallback failed: {str(e2)}")
                    return None
            else:
                print(f"Error downloading image from {image_url}: {str(e)}")
                return None

    def get_face_embedding(self, image_source: str, user_id: str = None) -> Optional[List[float]]:
        """
        Extract facial embedding from an image using DeepFace.
        
        Args:
            image_source: Local file path or URL of the image
            user_id: User ID for caching the image
            
        Returns:
            List of 128 floating point values representing the face embedding,
            or None if face extraction fails
        """
        # Determine an image path for DeepFace (prefers file paths)
        img_path: Optional[str] = None

        if os.path.isfile(image_source):
            img_path = image_source
        else:
            # Treat as URL and ensure it is cached to a deterministic path
            if not user_id:
                # If no user_id provided, derive a temporary cache name
                user_id = "TEMP"
            cache_filename = f"{user_id}.jpg"
            cache_path = os.path.join(self.cache_dir, cache_filename)

            image = self.download_image(image_source, user_id)
            if image is None:
                return None
            # Ensure the image is written to disk (download_image already tries when user_id is set)
            try:
                image.save(cache_path, 'JPEG', quality=95)
            except Exception:
                # If save fails, we cannot proceed with file-path based represent
                print(f"Warning: Failed to persist cached image at {cache_path}")
            img_path = cache_path

        # Extract facial embedding using DeepFace with a file path
        try:
            embedding_objs = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='opencv'
            )

            if embedding_objs and len(embedding_objs) > 0:
                embedding = embedding_objs[0]['embedding']
                return embedding
            else:
                print("No face detected in image")
                return None

        except Exception as e:
            print(f"Error extracting face embedding: {str(e)}")
            return None

    def get_face_embeddings_batch(self, users: List[dict]) -> List[Optional[List[float]]]:
        """
        Extract facial embeddings for multiple users in batch.
        
        Args:
            users: List of dicts with 'id' and 'profile_image_url' keys
            
        Returns:
            List of embedding vectors (same length as input, None for failed items)
        """
        embeddings = []
        
        for i, user in enumerate(users, 1):
            profile_image_url = user.get("profile_image_url")
            user_id = user.get("id")
            
            if profile_image_url:
                print(f"  Processing {i}/{len(users)}: {user.get('name', 'Unknown')}...")
                embedding = self.get_face_embedding(profile_image_url, user_id)
                embeddings.append(embedding)
            else:
                print(f"  Skipping {user.get('id')}: No profile image")
                embeddings.append(None)
        
        return embeddings

    def calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """
        Calculate cosine similarity between two facial embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1, where 1 = identical)
        """
        emb1_array = np.array(emb1)
        emb2_array = np.array(emb2)
        
        # Cosine similarity
        dot_product = np.dot(emb1_array, emb2_array)
        norm1 = np.linalg.norm(emb1_array)
        norm2 = np.linalg.norm(emb2_array)
        
        similarity = dot_product / (norm1 * norm2) if (norm1 * norm2) > 0 else 0
        return float(similarity)


def test_face_embedding():
    """Quick test of face embedding functionality."""
    helper = FaceEmbeddingHelper()
    
    # Test with a sample image URL
    test_url = "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400"
    print(f"\nTesting face embedding extraction...")
    embedding = helper.get_face_embedding(test_url, user_id="TEST-001")
    
    if embedding:
        print(f"✓ Successfully extracted face embedding")
        print(f"  Embedding dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
    else:
        print(f"✗ Failed to extract face embedding")


if __name__ == "__main__":
    test_face_embedding()
