"""
Query User Facial Similarity
Interactive interface to find users most similar by facial features.
Uses Pinecone vector search to measure and rank embedding distances.

EDUCATIONAL PURPOSE: Demonstrates vector similarity search fundamentals.
Educational context: Teaching how embeddings enable similarity measurement.
"""

import os
import json
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from dotenv import load_dotenv
from face_embedding_helper import FaceEmbeddingHelper

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "pclocal"))
index_name = "user-faces-index"

if not pc.has_index(index_name):
    print(f"Error: Index '{index_name}' does not exist.")
    print(f"Run 'python index_user_faces.py' first to create the index.")
    exit(1)

index = pc.Index(index_name)

# Load users
with open("data/store/users.json", "r") as f:
    users = json.load(f)

user_map = {user['id']: user for user in users}


class FacialSimilarityEngine:
    """Search for users with similar facial characteristics."""
    
    def __init__(self):
        self.face_helper = FaceEmbeddingHelper()
    
    def query_by_user_id(self, user_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find users most similar to a given user's facial features.
        
        Args:
            user_id: User ID to find similar users for (e.g., "USER-001")
            top_k: Number of similar users to return
            
        Returns:
            List of similar users ranked by facial similarity
        """
        user = user_map.get(user_id)
        if not user:
            print(f"User {user_id} not found")
            return []
        
        profile_image_url = user.get('profile_image_url')
        if not profile_image_url:
            print(f"User {user_id} has no profile image")
            return []
        
        # Get user's facial embedding
        user_embedding = self.face_helper.get_face_embedding(profile_image_url, user_id)
        if user_embedding is None:
            print(f"Could not extract facial embedding for {user_id}")
            return []
        
        # Query Pinecone for similar faces
        results = index.query(
            vector=user_embedding,
            top_k=top_k + 1,  # +1 to exclude self
            include_metadata=True
        )
        
        # Format and filter results (exclude the user themselves)
        similar_users = []
        for i, result in enumerate(results['matches']):
            # Skip self match
            if result['id'] == f"{user_id}-face":
                continue
            
            metadata = result['metadata']
            similar_users.append({
                'user_id': metadata['user_id'],
                'name': metadata['name'],
                'segment': metadata['segment'],
                'profile_image_url': metadata['profile_image_url'],
                'similarity_score': result['score'],
                'rank': len(similar_users) + 1
            })
            
            if len(similar_users) >= top_k:
                break
        
        return similar_users
    
    def query_by_image(self, image_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find users with facial features similar to a provided image.
        
        Args:
            image_path: Local file path or URL to the image
            top_k: Number of similar users to return
            
        Returns:
            List of similar users ranked by facial similarity
        """
        # Extract embedding from provided image
        face_embedding = self.face_helper.get_face_embedding(image_path)
        if face_embedding is None:
            print(f"Could not extract facial embedding from {image_path}")
            return []
        
        # Query Pinecone
        results = index.query(
            vector=face_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        similar_users = []
        for i, result in enumerate(results['matches']):
            metadata = result['metadata']
            similar_users.append({
                'user_id': metadata['user_id'],
                'name': metadata['name'],
                'segment': metadata['segment'],
                'profile_image_url': metadata['profile_image_url'],
                'similarity_score': result['score'],
                'rank': i + 1
            })
        
        return similar_users


def display_results(results: List[Dict[str, Any]], query_type: str = "facial similarity"):
    """Display similarity results in formatted table."""
    if not results:
        print(f"\nNo similar users found for your {query_type} query.")
        return
    
    print(f"\n{'─' * 100}")
    print(f"Top Results - {query_type.upper()}")
    print(f"{'─' * 100}")
    
    for result in results:
        print(f"\n{result['rank']}. {result['name']} ({result['user_id']})")
        print(f"   Segment: {result['segment']}")
        print(f"   Facial Similarity Score: {result['similarity_score']:.4f}")
        if result['profile_image_url']:
            print(f"   Profile: {result['profile_image_url']}")


def run_interactive_similarity():
    """Interactive facial similarity search interface."""
    print("\n" + "="*100)
    print("FACIAL SIMILARITY SEARCH".center(100))
    print("Find users with similar facial characteristics".center(100))
    print("="*100)
    print("\n⚠️  EDUCATIONAL PURPOSE: This demonstrates vector similarity measurement.")
    print("   It measures facial embedding distance for learning vector search concepts.")
    
    engine = FacialSimilarityEngine()
    
    while True:
        print("\n\nSearch Options:")
        print("1. Find users similar to a specific user (by ID)")
        print("2. Find users similar to an image (upload or provide path)")
        print("3. View all users")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            user_id = input("Enter user ID (e.g., USER-001): ").strip().upper()
            if user_id in user_map:
                print(f"\nSearching for users similar to {user_map[user_id]['name']}...")
                results = engine.query_by_user_id(user_id, top_k=5)
                display_results(results, "facial similarity to user")
            else:
                print(f"User {user_id} not found")
        
        elif choice == "2":
            image_path = input("Enter image path or URL: ").strip()
            if image_path:
                print("\nAnalyzing facial features and searching...")
                results = engine.query_by_image(image_path, top_k=5)
                display_results(results, "facial similarity to image")
        
        elif choice == "3":
            print("\n" + "─"*100)
            print("All Users".center(100))
            print("─"*100)
            for user in users:
                print(f"\n{user['id']}: {user['name']}")
                print(f"  Segment: {user['segment']}")
                print(f"  Preferred Categories: {', '.join(user['preferred_categories'])}")
        
        elif choice == "4":
            print("\nExiting facial similarity search. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    try:
        run_interactive_similarity()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
