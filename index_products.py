"""
Pinecone Product Indexing for Recommender Systems
Loads product data and creates a searchable vector index with rich metadata.
"""

import os
import time
from pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
from product_loader import prepare_products_for_indexing
from embedding_helper import EmbeddingHelper

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone with API key from environment variable
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "pclocal"))

# Initialize Embedding Helper
embedding_helper = EmbeddingHelper()

# Configuration
index_name = "product-recommender-index"
model_dimensions = 1536

# Delete index if it exists
if pc.has_index(index_name):
    pc.delete_index(index_name)
    print(f"Deleted existing index '{index_name}'.")
    time.sleep(1)

# Create a new index
pc.create_index(
    name=index_name,
    dimension=model_dimensions,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
print(f"Index '{index_name}' created.")

# Load and prepare products
print("\nLoading products from store data...")
products = prepare_products_for_indexing()
print(f"Loaded {len(products)} products")

# Generate embeddings for all products using batch processing
print("\nGenerating embeddings for products...")
product_texts = [p['text'] for p in products]
vectors = embedding_helper.get_embeddings_batch(product_texts)

print(f"Generated {len(vectors)} embeddings")

# Prepare vectors for upserting
index = pc.Index(index_name)
vectors_to_upsert = []

for i, product in enumerate(products):
    # Create vector record with product metadata
    vectors_to_upsert.append({
        "id": product['id'],
        "values": vectors[i],
        "metadata": product['metadata']
    })

# Upsert in batches of 100
batch_size = 100
for i in range(0, len(vectors_to_upsert), batch_size):
    batch = vectors_to_upsert[i:i + batch_size]
    try:
        index.upsert(vectors=batch)
        print(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")
        time.sleep(1)  # Wait 1 second between batches
    except Exception as e:
        print(f"Error upserting batch {i // batch_size + 1}: {e}")
        time.sleep(3)  # Wait longer before retrying
        try:
            index.upsert(vectors=batch)
            print(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors) on retry")
        except Exception as retry_error:
            print(f"Failed to upsert batch on retry: {retry_error}")

print(f"\nTotal product vectors upserted: {len(vectors_to_upsert)}")

# Display index stats
stats = index.describe_index_stats()
print(f"\nIndex Statistics:")
print(f"  Total vectors: {stats.total_vector_count}")
print(f"  Dimension: {stats.dimension}")

print("\nProduct index ready for recommender queries!")
print("Use query_products.py to search with personalized filters.")
