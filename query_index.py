import os
from pinecone import Pinecone
from dotenv import load_dotenv
from embedding_helper import EmbeddingHelper

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "pclocal"))

# Initialize Embedding Helper
embedding_helper = EmbeddingHelper()

# Configuration
index_name = "example-index"
top_k = 5  # Number of results to return

# Get the index
index = pc.Index(index_name)

def query_index(query_text: str, top_k: int = 5):
    """
    Query the Pinecone index with a text query.
    
    Args:
        query_text: The text to search for
        top_k: Number of top results to return
        
    Returns:
        List of matching results with metadata
    """
    # Generate embedding for the query
    print(f"\nGenerating embedding for query: '{query_text}'")
    query_vector = embedding_helper.get_embedding(query_text)
    
    # Search Pinecone
    print(f"Searching Pinecone index '{index_name}'...")
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    return results


def display_results(results):
    """Display search results in a formatted way."""
    if not results.matches:
        print("\nNo results found.")
        return
    
    print(f"\nFound {len(results.matches)} results:\n")
    
    for i, match in enumerate(results.matches, 1):
        metadata = match.metadata
        print(f"--- Result {i} (Score: {match.score:.4f}) ---")
        print(f"Title: {metadata.get('title', 'N/A')}")
        print(f"Company: {metadata.get('company', 'N/A')}")
        print(f"Year: {metadata.get('year', 'N/A')}")
        print(f"Location: {metadata.get('location', 'N/A')}")
        print(f"File: {metadata.get('fileName', 'N/A')}")
        print(f"Content:\n{metadata.get('content', 'N/A')}")
        print()

    # Brief summary of the top result
    best = results.matches[0]
    best_meta = best.metadata
    content_preview = best_meta.get('content', 'N/A')
    print("=== Best Match Summary ===")
    print(f"Title: {best_meta.get('title', 'N/A')}")
    print(f"Score: {best.score:.4f}")
    print(f"Company: {best_meta.get('company', 'N/A')}")
    print(f"Year: {best_meta.get('year', 'N/A')}")
    print(f"Location: {best_meta.get('location', 'N/A')}")
    print(f"File: {best_meta.get('fileName', 'N/A')}")
    print(f"Content:\n{content_preview}")


if __name__ == "__main__":
    print("Pinecone Query Interface")
    print("=" * 50)
    print(f"Index: {index_name}")
    print(f"Top K Results: {top_k}")
    print("=" * 50)
    
    while True:
        query = input("\nEnter your query (or 'quit' to exit): ").strip()
        
        if query.lower() == 'quit':
            print("Exiting...")
            break
        
        if not query:
            print("Please enter a valid query.")
            continue
        
        try:
            results = query_index(query, top_k)
            display_results(results)
        except Exception as e:
            print(f"Error querying index: {e}")
