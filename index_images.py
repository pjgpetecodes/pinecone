"""
Index Product Images for Multimodal Search (Offline using CLIP)
Generates image embeddings using local CLIP model and stores them alongside text embeddings.
Supports searching products by both text descriptions and images - completely offline!
"""

import os
import time
from pinecone import Pinecone
from dotenv import load_dotenv
from product_loader import load_products, load_orders, calculate_average_ratings, get_product_by_id
from image_helper import ImageHelper

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "pclocal"))
index_name = "product-recommender-index"

if not pc.has_index(index_name):
    print(f"Error: Index '{index_name}' does not exist. Run index_products.py first.")
    exit(1)

index = pc.Index(index_name)

# Initialize Image Helper (downloads CLIP model on first run)
print("Initializing CLIP model for offline image embeddings...")
image_helper = ImageHelper(target_dim=1536)  # Match Azure OpenAI embedding dimension
print(f"✓ CLIP initialized. Native: {image_helper.clip_dim}D, Padded to: {image_helper.target_dim}D")

# Load product data
print("\nLoading products...")
products = load_products()
orders = load_orders()
avg_ratings = calculate_average_ratings(products, orders)

print(f"Loaded {len(products)} products")

# Generate image embeddings and store as separate vectors
print("\nGenerating image embeddings using CLIP (offline)...")
print(f"Processing images - this may take a few minutes on first run (downloading model cache)...\n")

image_vectors = []
products_with_images = 0
failed_images = 0

# Process images
image_embeddings = image_helper.get_image_embeddings_batch(products)

for i, product in enumerate(products):
    embedding = image_embeddings[i]
    image_url = product.get('image_url')
    
    if embedding is not None:
        # Create metadata that indicates this is an image vector
        metadata = {
            'product_id': product['id'],
            'title': product['title'],
            'category': product['category'],
            'tags': product['tags'],
            'price': product['price'],
            'region': product['region'],
            'avg_rating': avg_ratings[product['id']],
            'created_at': product['created_at'],
            'image_url': image_url,
            'vector_type': 'image'  # Distinguish from text vectors
        }
        
        image_vectors.append({
            'id': f"{product['id']}-image",  # Suffix to distinguish from text vector
            'values': embedding,
            'metadata': metadata
        })
        
        products_with_images += 1
    else:
        print(f"  ⚠ {product['id']}: Failed to generate image embedding")
        failed_images += 1

print(f"\n✓ Generated embeddings for {products_with_images} product images")
if failed_images > 0:
    print(f"⚠ Failed: {failed_images}")

# Upsert image vectors in batches
if image_vectors:
    print(f"\nUpserting {len(image_vectors)} image vectors to Pinecone...")
    
    batch_size = 100
    for i in range(0, len(image_vectors), batch_size):
        batch = image_vectors[i:i + batch_size]
        try:
            index.upsert(vectors=batch)
            print(f"  ✓ Upserted batch {i // batch_size + 1} ({len(batch)} image vectors)")
            time.sleep(1)
        except Exception as e:
            print(f"  ✗ Error upserting batch {i // batch_size + 1}: {e}")
            time.sleep(3)
            try:
                index.upsert(vectors=batch)
                print(f"  ✓ Upserted batch {i // batch_size + 1} ({len(batch)} image vectors) on retry")
            except Exception as retry_error:
                print(f"  ✗ Failed to upsert batch {i // batch_size + 1}: {retry_error}")

print("\n" + "="*60)
print("Image indexing complete!")
print("="*60)
print(f"✓ Total image vectors stored: {products_with_images}")
print(f"✓ Total index size: {len(products) + products_with_images} vectors (text + image)")
print(f"✓ CLIP native dimension: {image_helper.clip_dim}D (padded to {image_helper.target_dim}D)")
print(f"✓ All processing done offline using CLIP")

