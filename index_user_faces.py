"""
Index User Profile Faces in Pinecone
Generates facial embeddings and stores them in a dedicated Pinecone index
for measuring and ranking facial similarity between users.

This is an EDUCATIONAL DEMONSTRATION of:
- Vector embedding generation from images
- Pinecone vector similarity search
- Embedding distance measurement

The system measures facial similarity for learning purposes only.
Not intended for real-world identification or user profiling.
"""

import os
import time
from pinecone import Pinecone
from dotenv import load_dotenv
from face_embedding_helper import FaceEmbeddingHelper

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "pclocal"))
index_name = "user-faces-index"

print("="*60)
print("FACIAL SIMILARITY INDEX SETUP".center(60))
print("="*60)

# Check if index exists, if not create it
if not pc.has_index(index_name):
    print(f"\n✓ Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=128,  # Facenet produces 128-dimensional embeddings
        metric="cosine",  # Cosine similarity for comparing faces
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )
    print(f"✓ Index '{index_name}' created")
    time.sleep(2)  # Wait for index to initialize
else:
    print(f"✓ Index '{index_name}' already exists")

index = pc.Index(index_name)

# Initialize Face Embedding Helper
print("\nInitializing face embedding model...")
face_helper = FaceEmbeddingHelper(model_name="Facenet")

# Load user data
print("\nLoading users...")
import json
with open("data/store/users.json", "r") as f:
    users = json.load(f)

print(f"Loaded {len(users)} users")

# Generate facial embeddings
print("\nGenerating facial embeddings using DeepFace (offline)...")
print(f"This may take a minute on first run (downloading models)...\n")

face_vectors = []
users_with_faces = 0
failed_faces = 0

# Process faces
face_embeddings = face_helper.get_face_embeddings_batch(users)

for i, user in enumerate(users):
    embedding = face_embeddings[i]
    
    if embedding is not None:
        # Create metadata for this user
        metadata = {
            'user_id': user['id'],
            'name': user['name'],
            'segment': user['segment'],
            'preferred_categories': user['preferred_categories'],
            'preferred_regions': user['preferred_regions'],
            'joined_at': user['joined_at'],
            'profile_image_url': user.get('profile_image_url', ''),
            'embedding_type': 'facial'
        }
        
        face_vectors.append({
            'id': f"{user['id']}-face",
            'values': embedding,
            'metadata': metadata
        })
        
        users_with_faces += 1
    else:
        print(f"  ⚠ {user['id']}: Failed to extract facial embedding")
        failed_faces += 1

print(f"\n✓ Generated embeddings for {users_with_faces} user faces")
if failed_faces > 0:
    print(f"⚠ Failed: {failed_faces}")

# Upsert to Pinecone
if face_vectors:
    print(f"\nUpserting {len(face_vectors)} facial vectors to Pinecone...")
    
    try:
        index.upsert(vectors=face_vectors)
        print(f"✓ Upserted {len(face_vectors)} facial vectors")
    except Exception as e:
        print(f"Error upserting: {e}")
        time.sleep(2)
        try:
            index.upsert(vectors=face_vectors)
            print(f"✓ Upserted {len(face_vectors)} facial vectors on retry")
        except Exception as retry_error:
            print(f"Failed to upsert: {retry_error}")

# Display summary
print("\n" + "="*60)
print("Face Index Ready!".center(60))
print("="*60)
print(f"✓ Total facial vectors stored: {users_with_faces}")
print(f"✓ Embedding model: Facenet")
print(f"✓ Embedding dimension: 128")
print(f"✓ Similarity metric: Cosine")
print(f"✓ All processing done OFFLINE (no API calls)")
print("\nNext: Run 'python query_user_similarity.py' to search!")
