"""
Multimodal Product Search using Pinecone
Supports searching by text description, image, or combination of both.
"""

import os
import sys
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from embedding_helper import EmbeddingHelper
from image_helper import ImageHelper
from product_loader import (
    load_products,
    load_users,
    load_orders,
    calculate_average_ratings,
    get_user_context,
    get_product_by_id
)


class MultimodalSearchEngine:
    """
    Search engine supporting text, image, and hybrid multimodal queries.
    """
    
    def __init__(self, index_name: str = "product-recommender-index"):
        load_dotenv()
        
        self.index_name = index_name
        self.embedding_helper = EmbeddingHelper()
        self.image_helper = ImageHelper()
        
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(self.index_name)
        
        # Load product data
        self.products = load_products()
        self.users = load_users()
        self.orders = load_orders()
        self.avg_ratings = calculate_average_ratings(self.products, self.orders)
    
    def query_by_text(self, query: str, top_k: int = 5, 
                     user_context: Optional[Dict] = None,
                     filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search using text query.
        
        Args:
            query: Text search query
            top_k: Number of results to return
            user_context: Optional user context for personalization
            filter_dict: Optional Pinecone filter metadata
            
        Returns:
            List of matching products
        """
        # Generate text embedding
        query_embedding = self.embedding_helper.get_embedding(query)
        
        # Search in Pinecone
        query_kwargs = {
            'vector': query_embedding,
            'top_k': top_k,
            'include_metadata': True
        }
        
        if filter_dict:
            query_kwargs['filter'] = filter_dict
        
        results = self.index.query(**query_kwargs)
        
        # Format results
        return self._format_search_results(results)
    
    def query_by_image(self, image_source: str, top_k: int = 5,
                      filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search using image query.
        
        Args:
            image_source: Local file path or URL of image to search with
            top_k: Number of results to return
            filter_dict: Optional Pinecone filter metadata
            
        Returns:
            List of matching products
        """
        # Generate image embedding
        image_embedding = self.image_helper.get_image_embedding(image_source)
        
        if image_embedding is None:
            print("Error: Could not generate embedding for image")
            return []
        
        # Search in Pinecone
        query_kwargs = {
            'vector': image_embedding,
            'top_k': top_k,
            'include_metadata': True
        }
        
        if filter_dict:
            query_kwargs['filter'] = filter_dict
        
        results = self.index.query(**query_kwargs)
        
        # Format results
        return self._format_search_results(results)
    
    def query_hybrid(self, text_query: str, image_source: Optional[str] = None,
                    text_weight: float = 0.6, top_k: int = 5,
                    filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Hybrid search combining text and image queries.
        
        Args:
            text_query: Text search query
            image_url: Optional image URL for combined search
            text_weight: Weight for text results (0-1), image gets 1-text_weight
            top_k: Number of results per modality
            filter_dict: Optional Pinecone filter metadata
            
        Returns:
            List of deduplicated matching products ranked by combined score
        """
        results_dict = {}
        
        # Text search
        text_results = self.query_by_text(text_query, top_k=top_k, filter_dict=filter_dict)
        for i, result in enumerate(text_results):
            product_id = result['id']
            text_score = 1.0 - (i / top_k)  # Normalize position to score
            
            if product_id not in results_dict:
                results_dict[product_id] = result.copy()
                results_dict[product_id]['_scores'] = {}
            
            results_dict[product_id]['_scores']['text'] = text_score
        
        # Image search if provided
        if image_source:
            image_weight = 1.0 - text_weight
            image_results = self.query_by_image(image_source, top_k=top_k, filter_dict=filter_dict)
            
            for i, result in enumerate(image_results):
                product_id = result['id']
                image_score = 1.0 - (i / top_k)  # Normalize position to score
                
                if product_id not in results_dict:
                    results_dict[product_id] = result.copy()
                    results_dict[product_id]['_scores'] = {}
                
                results_dict[product_id]['_scores']['image'] = image_score
        
        # Calculate combined scores
        for product_id in results_dict:
            scores = results_dict[product_id].get('_scores', {})
            text_score = scores.get('text', 0)
            image_score = scores.get('image', 0) if image_source else 0
            
            if image_source:
                combined_score = (text_score * text_weight) + (image_score * (1 - text_weight))
            else:
                combined_score = text_score
            
            results_dict[product_id]['combined_score'] = combined_score
            results_dict[product_id]['similarity'] = combined_score
        
        # Sort by combined score and return top k
        sorted_results = sorted(
            results_dict.values(),
            key=lambda x: x.get('combined_score', 0),
            reverse=True
        )[:top_k]
        
        # Clean up internal scores
        for result in sorted_results:
            result.pop('_scores', None)
        
        return sorted_results
    
    def _format_search_results(self, pinecone_results: Dict) -> List[Dict[str, Any]]:
        """
        Format Pinecone search results for display.
        
        Args:
            pinecone_results: Raw Pinecone query results
            
        Returns:
            Formatted results list
        """
        formatted = []
        
        for match in pinecone_results.get('matches', []):
            product_id = match['metadata']['product_id']
            product = get_product_by_id(self.products, product_id)
            
            result = {
                'id': product_id,
                'title': match['metadata']['title'],
                'description': product['description'] if product else '',
                'category': match['metadata']['category'],
                'price': match['metadata']['price'],
                'avg_rating': match['metadata']['avg_rating'],
                'image_url': match['metadata'].get('image_url', ''),
                'similarity': match['score']
            }
            
            formatted.append(result)
        
        return formatted


def display_results(results: List[Dict[str, Any]], query_type: str = "search"):
    """Display search results in formatted table."""
    if not results:
        print(f"\nNo products found for your {query_type}.")
        return
    
    print(f"\n{'─' * 120}")
    print(f"Top Results for {query_type.upper()}")
    print(f"{'─' * 120}")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   Category: {result['category']} | Price: ${result['price']:.2f} | Rating: {result['avg_rating']}/5.0")
        print(f"   Description: {result['description'][:100]}...")
        
        if result.get('image_url'):
            print(f"   Image: {result['image_url']}")
        
        print(f"   Similarity Score: {result.get('similarity', result.get('combined_score', 0)):.3f}")


def run_interactive_multimodal():
    """Interactive multimodal search CLI."""
    print("\n" + "=" * 120)
    print("MULTIMODAL PRODUCT SEARCH".center(120))
    print("Search by text, image, or combination of both".center(120))
    print("=" * 120)
    
    engine = MultimodalSearchEngine()
    
    while True:
        print("\n\nSearch Options:")
        print("1. Text search")
        print("2. Image search")
        print("3. Hybrid search (text + image)")
        print("4. Personalized text search (by user)")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            query = input("Enter your search query: ").strip()
            if query:
                results = engine.query_by_text(query)
                display_results(results, "text search")
        
        elif choice == "2":
            image_source = input("Enter image path or URL to search with: ").strip()
            if image_source:
                print("\nAnalyzing image and searching...")
                results = engine.query_by_image(image_source)
                display_results(results, "image search")
        
        elif choice == "3":
            text_query = input("Enter text query: ").strip()
            image_source = input("Enter image path or URL (or press Enter to skip): ").strip()
            
            if text_query:
                print("\nProcessing hybrid search...")
                results = engine.query_hybrid(
                    text_query=text_query,
                    image_source=image_source if image_source else None,
                    text_weight=0.6
                )
                display_results(results, "hybrid search")
        
        elif choice == "4":
            user_id = input("Enter user ID (e.g., USER-001): ").strip()
            query = input("Enter your search query: ").strip()
            
            user_context = get_user_context(user_id)
            if user_context and query:
                results = engine.query_by_text(query, user_context=user_context)
                display_results(results, f"personalized search for {user_context['name']}")
            elif not user_context:
                print("User not found.")
        
        elif choice == "5":
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    run_interactive_multimodal()
