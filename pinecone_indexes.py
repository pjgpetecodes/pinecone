import os
import time
from typing import List

from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from embedding_helper import EmbeddingHelper
from pdf_helper import PDFHelper
from sparse_vector_helper import SparseVectorHelper
from splade_helper import SpladeHelper


# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone with API key from environment variable
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "pclocal"))

# Initialize helpers
embedding_helper = EmbeddingHelper()
sparse_helper = SparseVectorHelper()
splade_helper = SpladeHelper()

# Configuration
index_name = "demo-hybrid-index"
model_dimensions = 1536  # Azure OpenAI Ada
import_dir = "import"
reset_index_on_start = os.getenv("RESET_INDEX_ON_START", "true").lower() == "true"  # default clean slate; set to false to keep data


def ensure_index(reset: bool = False):
    if reset and pc.has_index(index_name):
        print(f"Deleting existing index '{index_name}' for a clean run...")
        pc.delete_index(index_name)
        # brief pause to allow deletion to propagate
        time.sleep(1)

    if pc.has_index(index_name):
        return

    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=model_dimensions,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        print("Waiting for index to be ready...")
        time.sleep(1)
    print("Index ready.")


def ingest_pdf(pdf_path: str, use_splade: bool = True):
    pdf_helper = PDFHelper()
    extraction_result = pdf_helper.extract_paragraphs(pdf_path)

    print(f"\nExtracted {len(extraction_result.extracted_paragraphs)} paragraphs from {extraction_result.file_name}")
    print(f"Company: {extraction_result.company}, Year: {extraction_result.year}")

    # Namespace per company (lowercase, hyphenated)
    namespace = f"company-{extraction_result.company.lower()}"
    print(f"Using namespace: {namespace}")

    print("\nGenerating embeddings (dense)...")
    contents = [p.content for p in extraction_result.extracted_paragraphs]
    dense_vectors = embedding_helper.get_embeddings_batch(contents)

    if use_splade:
        print("Generating sparse vectors with SPLADE (may take a bit)...")
        sparse_vectors = [splade_helper.text_to_sparse(text) for text in contents]
    else:
        print("Generating sparse vectors (tf-based)...")
        sparse_vectors = [sparse_helper.text_to_sparse_tf(text) for text in contents]

    print(f"Generated {len(dense_vectors)} dense + {len(sparse_vectors)} sparse vectors")

    index = pc.Index(index_name)
    vectors_to_upsert = []

    for i, paragraph in enumerate(extraction_result.extracted_paragraphs):
        sparse = sparse_vectors[i]
        sparse_payload = {
            "indices": list(sparse.keys()),
            "values": list(sparse.values()),
        }

        vectors_to_upsert.append({
            "id": paragraph.id,
            "values": dense_vectors[i],
            "sparse_values": sparse_payload,
            "metadata": {
                "title": paragraph.title,
                "content": paragraph.content,
                "chunk_index": i,
                "chunk_location": paragraph.location,
                "company": extraction_result.company,
                "namespace": namespace,
                "year": extraction_result.year,
                "file_name": extraction_result.file_name,
                "source_type": "pdf",
                "ingest_ts": datetime.utcnow().isoformat() + "Z",
            }
        })

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        try:
            index.upsert(vectors=batch, namespace=namespace)
            print(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")
            time.sleep(1)
        except Exception as e:
            print(f"Error upserting batch {i // batch_size + 1}: {e}")
            time.sleep(3)
            try:
                index.upsert(vectors=batch, namespace=namespace)
                print(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors) on retry")
            except Exception as retry_error:
                print(f"Failed to upsert batch on retry: {retry_error}")

    print(f"Total vectors upserted for {extraction_result.company}: {len(vectors_to_upsert)}")


def find_pdfs(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        print(f"Import directory '{directory}' not found. Create it and add PDFs named <company>-<year>-file.pdf")
        return []
    pdfs = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(".pdf")]
    return sorted(pdfs)


if __name__ == "__main__":
    ensure_index(reset=reset_index_on_start)

    pdfs = find_pdfs(import_dir)
    if not pdfs:
        print("No PDFs found to ingest. Exiting.")
    else:
        print(f"Found {len(pdfs)} PDF(s) to ingest from '{import_dir}':")
        for p in pdfs:
            print(f" - {os.path.basename(p)}")
        for p in pdfs:
            ingest_pdf(p, use_splade=True)

    # List all indexes
    indexes = pc.list_indexes()
    print("\nIndexes:")
    for idx in indexes.indexes:
        print(f"  - {idx.name}")