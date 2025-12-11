import os
import time
from pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
from pdf_helper import PDFHelper
from embedding_helper import EmbeddingHelper

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone with API key from environment variable
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "pclocal"))

# Initialize Embedding Helper
embedding_helper = EmbeddingHelper()

# Configuration
index_name = "example-index"
model_dimensions = 1536
vector_search_profile_name = "my-vector-profile"
vector_search_config_name = "my-hnsw-config"

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

# Extract paragraphs from PDF
pdf_file = "XYZ-2021-Annual_Report.pdf"
pdf_helper = PDFHelper()
extraction_result = pdf_helper.extract_paragraphs(pdf_file)

print(f"\nExtracted {len(extraction_result.extracted_paragraphs)} paragraphs from {extraction_result.file_name}")
print(f"Company: {extraction_result.company}, Year: {extraction_result.year}")

# Generate embeddings for all paragraphs using batch processing
print("\nGenerating embeddings...")
contents = [p.content for p in extraction_result.extracted_paragraphs]
vectors = embedding_helper.get_embeddings_batch(contents)

print(f"Generated {len(vectors)} embeddings")

# Prepare vectors for upserting
index = pc.Index(index_name)
vectors_to_upsert = []

for i, paragraph in enumerate(extraction_result.extracted_paragraphs):
    # Create vector record with metadata in correct format
    vectors_to_upsert.append({
        "id": paragraph.id,
        "values": vectors[i],
        "metadata": {
            "title": paragraph.title,
            "content": paragraph.content,
            "location": paragraph.location,
            "company": extraction_result.company,
            "year": extraction_result.year,
            "fileName": extraction_result.file_name
        }
    })

# Upsert in batches of 100
batch_size = 100
for i in range(0, len(vectors_to_upsert), batch_size):
    batch = vectors_to_upsert[i:i + batch_size]
    try:
        index.upsert(vectors=batch)
        print(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")
        time.sleep(1)  # Wait 1 second between batches to avoid overwhelming the container
    except Exception as e:
        print(f"Error upserting batch {i // batch_size + 1}: {e}")
        time.sleep(3)  # Wait longer before retrying
        try:
            index.upsert(vectors=batch)
            print(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors) on retry")
        except Exception as retry_error:
            print(f"Failed to upsert batch on retry: {retry_error}")

print(f"\nTotal vectors upserted: {len(vectors_to_upsert)}")

# List all indexes
indexes = pc.list_indexes()
print("\nIndexes:")
for idx in indexes.indexes:
    print(f"  - {idx.name}")