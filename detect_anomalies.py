import argparse
import hashlib
from typing import List

from datetime import datetime
from sentence_transformers import SentenceTransformer

from generate_logs import generate_synthetic_logs
from log_event_schema import LogEvent
from pinecone_logs_index import get_pinecone, ensure_index, upsert_vectors, INDEX_NAME


def embed_events(model: SentenceTransformer, events: List[LogEvent]) -> List[List[float]]:
    texts = [e.to_text() for e in events]
    return model.encode(texts, normalize_embeddings=True).tolist()


def make_id(e: LogEvent) -> str:
    raw = f"{e.timestamp.isoformat()}|{e.level}|{e.service}|{e.message}|{e.trace_id}|{e.user_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Embed logs and detect anomalies via Pinecone")
    parser.add_argument("--count", type=int, default=300, help="Total synthetic logs to generate")
    parser.add_argument("--anomaly_ratio", type=float, default=0.15, help="Fraction of anomalies")
    parser.add_argument("--top_k", type=int, default=10, help="Neighbors to inspect for scoring")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible results")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    import random
    random.seed(args.seed)

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Generating synthetic logs...")
    events = generate_synthetic_logs(n=args.count, anomaly_ratio=args.anomaly_ratio)
    vectors = embed_events(model, events)

    print("Connecting to Pinecone and ensuring index...")
    pc = get_pinecone()
    ensure_index(pc, dimension=len(vectors[0]), metric="cosine")

    print("Upserting vectors...")
    items = []
    for e, v in zip(events, vectors):
        items.append((make_id(e), v, e.to_metadata()))
    upsert_vectors(pc, items)

    index = pc.Index(INDEX_NAME)
    print("Scoring anomalies via neighbor distances...")
    # Simple anomaly score: 1 - average top_k similarity
    scores = []
    for e, v in zip(events, vectors):
        res = index.query(vector=v, top_k=args.top_k, include_metadata=False)
        sims = [m["score"] for m in res["matches"] if m["id"] != make_id(e)]
        if not sims:
            avg_sim = 0.0
        else:
            avg_sim = sum(sims) / len(sims)
        scores.append((avg_sim, e))

    # Sort by average similarity
    scores.sort(key=lambda x: x[0])
    
    # Categorize events - scan all to find predefined anomalies
    from generate_logs import USER_PROFILES
    true_anomalies = []
    vector_outliers = []
    
    # First pass: find all true anomalies (our predefined ones)
    for avg_sim, e in scores:
        is_true_anomaly = False
        
        # Check if it's a predefined anomaly by trace ID
        if e.trace_id and e.trace_id.startswith("anomaly-"):
            is_true_anomaly = True
        # Fallback checks
        elif e.attributes and e.attributes.get("amount", 0) > 5000:
            is_true_anomaly = True
        elif "high-risk location" in e.message:
            is_true_anomaly = True
        elif "out-of-pattern" in e.message:
            is_true_anomaly = True
        
        if is_true_anomaly:
            true_anomalies.append((avg_sim, e))
    
    # Second pass: find vector outliers from low-similarity events
    for avg_sim, e in scores[:30]:  # Check top 30 lowest similarity
        # Skip if already in true anomalies
        if any(e.trace_id == ta[1].trace_id for ta in true_anomalies):
            continue
            
        # Check if it's a vector outlier (in-pattern but low similarity)
        if e.user_id in USER_PROFILES and e.attributes and e.attributes.get("category"):
            user_profile = USER_PROFILES[e.user_id]
            purchase_category = e.attributes.get("category")
            if purchase_category in user_profile["categories"]:
                vector_outliers.append((avg_sim, e))
    
    # Sort true anomalies by similarity (lowest first)
    true_anomalies.sort(key=lambda x: x[0])
    
    print("\n" + "="*80)
    print("MOST NORMAL EVENTS (High Similarity to Neighbors)")
    print("="*80)
    print("These logs have patterns similar to many other events in the dataset.\n")
    
    for i, (avg_sim, e) in enumerate(scores[-10:][::-1], start=1):
        print(f"{i:02d}. Similarity: {avg_sim:.3f}")
        print(f"    {e.to_text()}")
        print()
    
    print("\n" + "="*80)
    print("TRUE ANOMALIES (Behavioral Red Flags)")
    print("="*80)
    print("These events represent genuine security or fraud concerns:\n")
    
    for i, (avg_sim, e) in enumerate(true_anomalies, start=1):
        print(f"{i:02d}. Anomaly Score: {1.0 - avg_sim:.3f} (Similarity: {avg_sim:.3f})")
        
        # Identify what makes it anomalous
        if e.attributes and e.attributes.get("amount", 0) > 5000:
            amount = e.attributes.get("amount")
            product = e.attributes.get("product", "unknown item")
            print(f"    💰 UNUSUAL AMOUNT: ${amount:,.2f} purchase")
            print(f"    Normal transactions are $50-$300")
            print(f"    Product: {product}")
        
        elif "high-risk location" in e.message and e.attributes:
            country = e.attributes.get("country")
            print(f"    🌍 UNUSUAL LOGIN LOCATION: {country}")
            print(f"    User typically logs in from US, CA, UK, DE, FR, AU")
            print(f"    This location flagged as high-risk")
        
        elif "out-of-pattern" in e.message and e.attributes:
            profile_type = e.attributes.get("user_profile", "user")
            product = e.attributes.get("product", "item")
            category = e.attributes.get("category", "unknown")
            typical = e.attributes.get("typical_categories", "normal items")
            print(f"    🛒 UNUSUAL PURCHASE CATEGORY: {profile_type} bought '{product}'")
            print(f"    Category: {category}")
            print(f"    User normally buys: {typical}")
        
        print(f"    {e.to_text()}")
        print()
    
    if vector_outliers:
        print("\n" + "="*80)
        print("VECTOR OUTLIERS (Low Similarity but Behaviorally Normal)")
        print("="*80)
        print("These have low vector similarity but match user purchase patterns.\n")
        print("Often caused by sparse data in certain product categories:\n")
        
        for i, (avg_sim, e) in enumerate(vector_outliers[:5], start=1):
            print(f"{i:02d}. Similarity: {avg_sim:.3f}")
            user_profile = USER_PROFILES[e.user_id]
            purchase_category = e.attributes.get("category")
            product = e.attributes.get("product", "item")
            print(f"    📊 {user_profile['name']} bought '{product}' ({purchase_category}) ✓")
            print(f"    This matches their profile: {', '.join(user_profile['categories'])}")
            print(f"    {e.to_text()}")
            print()


if __name__ == "__main__":
    main()
