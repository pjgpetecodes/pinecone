import os
from pinecone import Pinecone
from dotenv import load_dotenv
from embedding_helper import EmbeddingHelper
from sparse_vector_helper import SparseVectorHelper
from splade_helper import SpladeHelper

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "pclocal"))

# Initialize helpers
embedding_helper = EmbeddingHelper()
sparse_helper = SparseVectorHelper()
splade_helper = SpladeHelper()

# Configuration
index_name = "example-index"
top_k = 5  # Number of results to return

# Get the index
index = pc.Index(index_name)


def prompt_namespace() -> str:
    company = input("Enter company name (e.g., acme): ").strip().lower()
    return f"company-{company}"


def prompt_query() -> str:
    return input("Enter your query (or 'quit' to return to menu): ").strip()

def hybrid_query(query_text: str, namespace: str, top_k: int = 5, mode: str = "hybrid", alpha: float = 0.7, use_splade: bool = True, metadata_filter: dict | None = None):
    """
    Query Pinecone with dense, sparse, or hybrid mode.

    Args:
        query_text: user query
        namespace: company namespace (e.g., company-acme)
        top_k: number of results
        mode: 'dense' | 'sparse' | 'hybrid'
        alpha: blend weight for dense vs sparse when hybrid (0-1)
    """
    dense_vec = embedding_helper.get_embedding(query_text)
    if use_splade:
        sparse = splade_helper.text_to_sparse(query_text)
    else:
        sparse = sparse_helper.text_to_sparse_tf(query_text)
    sparse_payload = {"indices": list(sparse.keys()), "values": list(sparse.values())}

    if mode == "dense":
        q = {"vector": dense_vec}
    elif mode == "sparse":
        q = {"sparse_vector": sparse_payload}
    else:  # hybrid
        # scale dense vector by alpha; scale sparse by (1-alpha) via values
        scaled_dense = [v * alpha for v in dense_vec]
        scaled_sparse = {"indices": sparse_payload["indices"], "values": [v * (1 - alpha) for v in sparse_payload["values"]]}
        q = {"vector": scaled_dense, "sparse_vector": scaled_sparse}

    query_kwargs = {
        "top_k": top_k,
        "include_metadata": True,
        "namespace": namespace,
        **q,
    }

    if metadata_filter:
        query_kwargs["filter"] = metadata_filter

    results = index.query(**query_kwargs)
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


def run_single_query(mode: str, use_splade: bool, alpha: float):
    namespace = prompt_namespace()
    while True:
        query = prompt_query()
        if query.lower() == "quit":
            return
        if not query:
            print("Please enter a valid query.")
            continue
        try:
            results = hybrid_query(
                query,
                namespace=namespace,
                top_k=top_k,
                mode=mode,
                alpha=alpha,
                use_splade=use_splade,
            )
            display_results(results)
        except Exception as e:
            print(f"Error querying index: {e}")


def show_menu():
    print("\nDemo Menu (select a scenario):")
    print(" 1) Dense-only (baseline)")
    print(" 2) Sparse-only (lexical)")
    print(" 3) Hybrid (blend dense+sparse)")
    print(" 4) Namespace isolation demo")
    print(" 5) Dimensions explanation")
    print(" 6) Metadata fields demo")
    print(" q) Quit")


def namespace_demo():
    print("\nNamespace demo: results are scoped per company.")
    print("Use the same query across two namespaces to see different corpora.")
    print("Tip: ingest multiple PDFs named <company>-<year>-file.pdf into 'import/' then run pinecone_indexes.py")
    ns = prompt_namespace()
    query = prompt_query()
    if query.lower() == "quit" or not query:
        return
    try:
        results = hybrid_query(query, namespace=ns, top_k=top_k, mode="hybrid", alpha=0.7, use_splade=True)
        display_results(results)
    except Exception as e:
        print(f"Error querying index: {e}")
    print("Run again with a different company to contrast.")


def dimensions_note():
    print("\nDimensions: embeddings are 1536-dim (Ada).")
    print("Index must match the embedding dimension set in pinecone_indexes.py (model_dimensions).")
    print("Higher dims = more storage/latency but richer signal; lower dims = lighter but potential recall loss.")


def metadata_demo():
    print("\nMetadata demo: showing returned metadata fields (title, company, year, chunk_location, file_name, source_type, ingest_ts).")
    ns = prompt_namespace()
    query = prompt_query()
    if query.lower() == "quit" or not query:
        return
    metadata_filter = prompt_metadata_filter()
    try:
        results = hybrid_query(query, namespace=ns, top_k=top_k, mode="hybrid", alpha=0.7, use_splade=True, metadata_filter=metadata_filter)
        if not results.matches:
            print("No results found.")
            return
        first = results.matches[0]
        print("\nFirst result metadata:")
        for k, v in first.metadata.items():
            print(f"- {k}: {v}")
        display_results(results)
    except Exception as e:
        print(f"Error querying index: {e}")


def prompt_metadata_filter() -> dict | None:
    print("\nAdd metadata filters (press Enter to skip any field):")
    year_min = input("Minimum year (e.g., 2022): ").strip()
    year_max = input("Maximum year (optional): ").strip()
    source_type = input("Source type (e.g., pdf): ").strip()
    company = input("Company (exact match, e.g., ACME): ").strip()
    file_name = input("File name contains (substring match not supported; leave blank unless exact): ").strip()

    filt: dict = {}
    if year_min.isdigit():
        filt.setdefault("year", {})["$gte"] = int(year_min)
    if year_max.isdigit():
        filt.setdefault("year", {})["$lte"] = int(year_max)
    if source_type:
        filt["source_type"] = {"$eq": source_type}
    if company:
        filt["company"] = {"$eq": company}
    if file_name:
        filt["file_name"] = {"$eq": file_name}

    if not filt:
        print("No filter applied.")
        return None

    print(f"Applying filter: {filt}")
    return filt


if __name__ == "__main__":
    print("Pinecone Query Demo")
    print("=" * 50)
    print(f"Index: {index_name}")

    while True:
        show_menu()
        choice = input("Select option: ").strip().lower()

        if choice == "q":
            print("Exiting demo...")
            break
        elif choice == "1":
            run_single_query(mode="dense", use_splade=False, alpha=1.0)
        elif choice == "2":
            run_single_query(mode="sparse", use_splade=True, alpha=0.0)
        elif choice == "3":
            alpha_str = input("Blend weight for dense (0.0-1.0) [0.7]: ").strip()
            alpha = float(alpha_str) if alpha_str else 0.7
            alpha = max(0.0, min(1.0, alpha))
            use_splade_str = input("Use SPLADE for sparse? [Y/n]: ").strip().lower()
            use_splade = False if use_splade_str == "n" else True
            run_single_query(mode="hybrid", use_splade=use_splade, alpha=alpha)
        elif choice == "4":
            namespace_demo()
        elif choice == "5":
            dimensions_note()
        elif choice == "6":
            metadata_demo()
        else:
            print("Invalid choice. Try again.")
