"""
Index Product Images for Multimodal Search
Generates image embeddings and stores them alongside text embeddings.
Supports searching products by both text descriptions and images.
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

# Initialize Image Helper
image_helper = ImageHelper()

# Load product data
print("Loading products...")
products = load_products()
orders = load_orders()
avg_ratings = calculate_average_ratings(products, orders)

print(f"Loaded {len(products)} products")

# Generate image embeddings and store as separate vectors
print("\nGenerating image embeddings...")

image_vectors = []
products_with_images = 0
failed_images = 0

for product in products:
    image_url = product.get('image_url')
    
    if not image_url:
        print(f"  {product['id']}: No image URL")
        continue
    
    print(f"  Processing {product['id']}: {product['title']}...")
    
    try:
        # Generate image embedding
        image_embedding = image_helper.get_image_embedding(image_url, product['description'])
        
        if image_embedding:
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
                'values': image_embedding,
                'metadata': metadata
            })
            
            products_with_images += 1
            time.sleep(0.5)  # Rate limiting for API calls
        else:
            print(f"  {product['id']}: Failed to generate image embedding")
            failed_images += 1
    
    except Exception as e:
        print(f"  {product['id']}: Error - {str(e)}")
        failed_images += 1

print(f"\nGenerated embeddings for {products_with_images} product images")
print(f"Failed: {failed_images}")

# Upsert image vectors in batches
if image_vectors:
    print(f"\nUpserting {len(image_vectors)} image vectors...")
    
    batch_size = 50  # Smaller batch size for image vectors due to API costs
    for i in range(0, len(image_vectors), batch_size):
        batch = image_vectors[i:i + batch_size]
        try:
            index.upsert(vectors=batch)
            print(f"  Upserted batch {i // batch_size + 1} ({len(batch)} image vectors)")
            time.sleep(2)  # Longer wait between batches to avoid rate limiting
        except Exception as e:
            print(f"  Error upserting batch {i // batch_size + 1}: {e}")
            time.sleep(5)
            try:
                index.upsert(vectors=batch)
                print(f"  Upserted batch {i // batch_size + 1} ({len(batch)} image vectors) on retry")
            except Exception as retry_error:
                print(f"  Failed to upsert batch {i // batch_size + 1}: {retry_error}")

print("\nImage indexing complete!")
print(f"Total image vectors stored: {products_with_images}")
print(f"Total index size: {len(products) + products_with_images} vectors (text + image)")
