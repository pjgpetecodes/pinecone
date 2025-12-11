"""
Personalized Product Query with Pinecone
Demonstrates metadata filtering and personalized recommendations.
"""

import os
from typing import Optional
from pinecone import Pinecone
from dotenv import load_dotenv
from embedding_helper import EmbeddingHelper
from product_loader import get_user_context

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "pclocal"))
embedding_helper = EmbeddingHelper()

# Configuration
index_name = "product-recommender-index"


def build_metadata_filter(user_context: dict = None, 
                         category: str = None, 
                         region: str = None,
                         min_price: float = None,
                         max_price: float = None) -> dict:
    """
    Build a Pinecone metadata filter based on user preferences and query parameters.
    
    Args:
        user_context: User context with preferences
        category: Specific category to filter
        region: Specific region to filter
        min_price: Minimum price filter
        max_price: Maximum price filter
        
    Returns:
        Pinecone filter dictionary
    """
    filters = {}
    
    # Category filter (from user preference or explicit)
    if category:
        filters['category'] = {'$eq': category}
    elif user_context and user_context.get('preferred_categories'):
        # Filter by user's preferred categories
        filters['category'] = {'$in': user_context['preferred_categories']}
    
    # Region filter
    if region:
        filters['region'] = {'$eq': region}
    elif user_context and user_context.get('preferred_regions'):
        filters['region'] = {'$in': user_context['preferred_regions']}
    
    # Price range filter
    if min_price is not None or max_price is not None:
        price_filter = {}
        if min_price is not None:
            price_filter['$gte'] = min_price
        if max_price is not None:
            price_filter['$lte'] = max_price
        filters['price'] = price_filter
    
    return filters if filters else None


def query_products(query_text: Optional[str], 
                  user_id: str = None,
                  category: str = None,
                  region: str = None,
                  min_price: float = None,
                  max_price: float = None,
                  top_k: int = 5) -> list:
    """
    Query products with personalization and filtering.
    
    Args:
        query_text: Natural language search query
        user_id: User ID for personalized results
        category: Filter by category
        region: Filter by region
        min_price: Minimum price
        max_price: Maximum price
        top_k: Number of results to return
        popularity_boost: Whether to boost popular items
        
    Returns:
        List of matching products with scores
    """
    # Get user context if user_id provided
    user_context = None
    if user_id:
        user_context = get_user_context(user_id)
        if user_context:
            print(f"\n🔍 Personalizing results for: {user_context['name']} ({user_context['segment']})")
            print(f"   Preferred categories: {', '.join(user_context['preferred_categories'])}")
            print(f"   Preferred regions: {', '.join(user_context['preferred_regions'])}")
            print(f"   Previous purchases: {user_context['order_count']} orders")

    # If no query provided, seed from last purchase (if available)
    query_seeded_from_last_order = False
    last_product_id = None
    if not query_text:
        if user_context and user_context.get('last_order'):
            last = user_context['last_order']
            last_product_id = last['product_id']
            query_text = (
                f"Product: {last['title']}\n"
                f"Description: {last['description']}\n"
                f"Category: {last['category']}\n"
                f"Tags: {', '.join(last.get('tags', []))}\n"
                f"Region: {last['region']}"
            )
            query_seeded_from_last_order = True
            print(f"\n🆕 Recommendations based on last purchase: {last['title']} ({last['product_id']})")
        else:
            print("\nNo query provided and no orders found for this user. Please enter a query or choose a user with purchase history.")
            return []
    
    # Build metadata filter
    # Skip user preferences when seeding from last purchase to get truly similar items
    metadata_filter = build_metadata_filter(
        user_context=None if query_seeded_from_last_order else user_context,
        category=category,
        region=region,
        min_price=min_price,
        max_price=max_price
    )
    
    if metadata_filter:
        print(f"\n🎯 Applied filters: {metadata_filter}")
    
    # Generate query embedding
    query_vector = embedding_helper.get_embedding(query_text)
    
    # Query Pinecone
    index = pc.Index(index_name)
    
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter
    )

    # Remove the exact last purchased product from recommendations
    if last_product_id and results.matches:
        results.matches = [m for m in results.matches if m.metadata.get('product_id') != last_product_id]
    
    return results.matches


def display_results(matches: list, user_context: dict = None):
    """
    Display query results in a formatted way.
    
    Args:
        matches: List of query matches
        user_context: Optional user context for highlighting purchased items
    """
    purchased_ids = set(user_context['purchased_products']) if user_context else set()
    
    print(f"\n{'='*80}")
    print(f"🛍️  TOP RECOMMENDATIONS ({len(matches)} results)")
    print(f"{'='*80}\n")
    
    for i, match in enumerate(matches, 1):
        metadata = match.metadata
        is_purchased = metadata['product_id'] in purchased_ids
        
        print(f"{i}. {metadata['title']}")
        print(f"   Category: {metadata['category']} | Region: {metadata['region']}")
        print(f"   Price: ${metadata['price']} | Avg Rating: {metadata['avg_rating']}/5.0")
        print(f"   Score: {match.score:.4f}")
        
        if is_purchased:
            print(f"   ✅ Previously purchased")
        
        # Show tags
        if metadata.get('tags'):
            tags_str = ', '.join(metadata['tags'])
            print(f"   Tags: {tags_str}")
        
        print()


def run_guided_tour():
    """
    Run a short guided tour with real-world scenarios.
    """
    print("\n" + "="*80)
    print("GUIDED TOUR: Everyday Recommender Scenarios")
    print("="*80)
    
    print("""
This tour demonstrates four real-world personalized recommendation scenarios:

  1️⃣  Just Purchased → You Might Also Like
      After a purchase, suggest related items without re-showing the bought product.
      
  2️⃣  Welcome Back → Because You Bought Before
      Logged-in user: use preferences and history to suggest next picks.
      
  3️⃣  Active Search → Closest Match + Related
      User searches for something; we combine similarity, preferences & popularity.
      
  4️⃣  Also Popular in Your Crowd
      Show segment overlap: fitness enthusiasts discovering shared purchases.

Each scenario uses different users, queries, and filters to showcase the full 
power of metadata filtering, personalization, and vector similarity.
""")
    
    input("Press Enter to begin the tour...")

    scenarios = [
        {
            "title": "Just purchased → You might also like",
            "description": "User just bought something; suggest related items without re-showing the purchase.",
            "query": None,  # seed from last order
            "user_id": "USER-002",  # tech-professional
            "category": None,
            "region": None
        },
        {
            "title": "Welcome back → Because you bought before",
            "description": "Logged-in user, use preferences and history to suggest next picks.",
            "query": "kitchen essentials",
            "user_id": "USER-005",  # home-chef
            "category": None,
            "region": None
        },
        {
            "title": "Active search → Closest match + related",
            "description": "User searches; we combine similarity, prefs, and popularity.",
            "query": "wireless headphones",
            "user_id": "USER-010",  # tech-professional (new)
            "category": None,
            "region": None
        },
        {
            "title": "Also popular in your crowd",
            "description": "Show segment overlap (fitness enthusiasts share buys).",
            "query": "fitness gear",
            "user_id": "USER-009",  # fitness-enthusiast (new)
            "category": None,
            "region": None
        }
    ]

    for scenario in scenarios:
        print(f"\n{'-'*80}")
        print(f"{scenario['title']}")
        print(f"{scenario['description']}")
        print(f"User: {scenario['user_id']}")
        if scenario['query']:
            print(f"Query: {scenario['query']}")
        else:
            print("Query: (skipped, using last purchase)")

        matches = query_products(
            query_text=scenario['query'],
            user_id=scenario['user_id'],
            category=scenario['category'],
            region=scenario['region'],
            top_k=5
        )

        user_context = get_user_context(scenario['user_id'])
        display_results(matches, user_context)

        input("\nPress Enter for the next scenario...")

    print("\n" + "="*80)
    print("End of guided tour. Switching to interactive mode...")
    print("="*80)


def run_interactive_query():
    """
    Run an interactive query session.
    """
    print("\n" + "="*80)
    print("🛍️  PERSONALIZED PRODUCT RECOMMENDER")
    print("="*80)
    print("\nAvailable Users:")
    print("  USER-001: Alice (fitness-enthusiast)")
    print("  USER-002: Bob (tech-professional)")
    print("  USER-003: Carol (eco-conscious)")
    print("  USER-004: David (wellness-seeker)")
    print("  USER-005: Emma (home-chef)")
    print("  USER-006: Frank (remote-worker)")
    print("  USER-007: Grace (outdoor-adventurer)")
    print("  USER-008: Henry (gadget-lover)")
    print("  USER-009: Isabella (fitness-enthusiast)")
    print("  USER-010: Jason (tech-professional)")
    
    while True:
        print("\n" + "-"*80)
        query = input("\n🔍 Enter search query (press Enter to skip and use last purchase, or type 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the Product Recommender!")
            break

        # Ask for user ID for personalization (needed for no-query mode)
        user_id = input("👤 Enter User ID for personalization (or press Enter to skip): ").strip().upper()
        if user_id and not user_id.startswith("USER-"):
            user_id = None
        
        # If no query and no user, prompt again
        if not query and not user_id:
            print("Please provide a query or a user ID with purchase history.")
            continue
        
        # Optional filters
        category = input("📁 Filter by category (or press Enter to skip): ").strip()
        category = category if category else None
        
        region = input("🌍 Filter by region (or press Enter to skip): ").strip()
        region = region if region else None
        
        # Query products
        try:
            matches = query_products(
                query_text=query,
                user_id=user_id if user_id else None,
                category=category,
                region=region,
                top_k=5
            )
            
            # Get user context for display
            user_context = get_user_context(user_id) if user_id else None
            
            # Display results
            display_results(matches, user_context)
            
            if not matches:
                print("No products found matching your criteria. Try different filters or query.")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Make sure the product index has been created (run index_products.py first)")


if __name__ == "__main__":
    # Check if running interactively or with example queries
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        # Run interactive mode
        run_interactive_query()

    elif len(sys.argv) > 1 and sys.argv[1] == "--examples":
        print("\n" + "="*80)
        print("EXAMPLE QUERIES")
        print("="*80)
        
        examples = [
            {
                "query": "wireless headphones",
                "user_id": "USER-002",
                "description": "Tech professional looking for audio gear"
            },
            {
                "query": "workout equipment",
                "user_id": "USER-001",
                "description": "Fitness enthusiast shopping for exercise gear"
            },
            {
                "query": "eco-friendly kitchen products",
                "user_id": "USER-003",
                "description": "Eco-conscious shopper looking for sustainable items"
            },
            {
                "query": "office accessories",
                "user_id": "USER-006",
                "description": "Remote worker setting up home office"
            }
        ]
        
        for example in examples:
            print(f"\n{'='*80}")
            print(f"Example: {example['description']}")
            print(f"Query: '{example['query']}'")
            print(f"{'='*80}")
            
            matches = query_products(
                query_text=example['query'],
                user_id=example['user_id'],
                top_k=3
            )
            
            user_context = get_user_context(example['user_id'])
            display_results(matches, user_context)
            
            input("\nPress Enter to continue to next example...")

    else:
        # Run guided tour by default
        run_guided_tour()
        run_interactive_query()
