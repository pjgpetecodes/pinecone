"""
Product Loader for Recommender Systems
Loads product data from JSON and prepares it for embedding and vector storage.
"""

import json
import csv
import uuid
from datetime import datetime
from typing import List, Dict, Any


def load_products(file_path: str = "data/store/products.json") -> List[Dict[str, Any]]:
    """
    Load products from JSON file.
    
    Args:
        file_path: Path to products JSON file
        
    Returns:
        List of product dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        products = json.load(f)
    return products


def get_product_by_id(products: List[Dict[str, Any]], product_id: str) -> Dict[str, Any] | None:
    """
    Find a product by ID from a loaded product list.
    """
    for product in products:
        if product.get('id') == product_id:
            return product
    return None


def load_users(file_path: str = "data/store/users.json") -> Dict[str, Dict[str, Any]]:
    """
    Load users from JSON file.
    
    Args:
        file_path: Path to users JSON file
        
    Returns:
        Dictionary mapping user_id to user data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        users = json.load(f)
    return {user['id']: user for user in users}


def load_orders(file_path: str = "data/store/orders.csv") -> List[Dict[str, Any]]:
    """
    Load orders from CSV file.
    
    Args:
        file_path: Path to orders CSV file
        
    Returns:
        List of order dictionaries
    """
    orders = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            orders.append(row)
    return orders


def calculate_popularity_scores(products: List[Dict[str, Any]], 
                                orders: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate updated popularity scores based on recent orders.
    
    Args:
        products: List of products
        orders: List of orders
        
    Returns:
        Dictionary mapping product_id to updated popularity score
    """
    # Count orders per product
    order_counts = {}
    for order in orders:
        product_id = order['product_id']
        rating = float(order['rating'])
        if product_id not in order_counts:
            order_counts[product_id] = {'count': 0, 'total_rating': 0}
        order_counts[product_id]['count'] += 1
        order_counts[product_id]['total_rating'] += rating
    
    # Update popularity: base + (order_count * avg_rating / 2)
    popularity_scores = {}
    for product in products:
        product_id = product['id']
        base_popularity = product['popularity']
        
        if product_id in order_counts:
            count = order_counts[product_id]['count']
            avg_rating = order_counts[product_id]['total_rating'] / count
            boost = (count * avg_rating) / 2
            popularity_scores[product_id] = min(10.0, base_popularity + boost)
        else:
            popularity_scores[product_id] = base_popularity
    
    return popularity_scores


def prepare_product_for_embedding(product: Dict[str, Any], 
                                  popularity_score: float) -> tuple[str, Dict[str, Any]]:
    """
    Prepare a product for embedding by creating searchable text and metadata.
    
    Args:
        product: Product dictionary
        popularity_score: Updated popularity score
        
    Returns:
        Tuple of (embedding_text, metadata_dict)
    """
    # Create rich text representation for embedding
    embedding_text = f"""
Product: {product['title']}
Description: {product['description']}
Category: {product['category']}
Tags: {', '.join(product['tags'])}
Price: ${product['price']}
Region: {product['region']}
""".strip()
    
    # Prepare metadata for filtering and display
    metadata = {
        'product_id': product['id'],
        'title': product['title'],
        'category': product['category'],
        'tags': product['tags'],
        'price': product['price'],
        'region': product['region'],
        'popularity': round(popularity_score, 2),
        'created_at': product['created_at']
    }
    
    return embedding_text, metadata


def prepare_products_for_indexing(products_file: str = "data/store/products.json",
                                  orders_file: str = "data/store/orders.csv") -> List[Dict[str, Any]]:
    """
    Load products and orders, calculate popularity, and prepare for indexing.
    
    Args:
        products_file: Path to products JSON
        orders_file: Path to orders CSV
        
    Returns:
        List of dictionaries with id, text, and metadata for each product
    """
    # Load data
    products = load_products(products_file)
    orders = load_orders(orders_file)
    
    # Calculate updated popularity
    popularity_scores = calculate_popularity_scores(products, orders)
    
    # Prepare for indexing
    indexed_products = []
    for product in products:
        text, metadata = prepare_product_for_embedding(product, popularity_scores[product['id']])
        
        indexed_products.append({
            'id': product['id'],  # Use product ID as vector ID
            'text': text,
            'metadata': metadata
        })
    
    return indexed_products


def get_user_context(user_id: str, 
                    users_file: str = "data/store/users.json",
                    orders_file: str = "data/store/orders.csv") -> Dict[str, Any]:
    """
    Get user context including preferences and purchase history.
    
    Args:
        user_id: User ID
        users_file: Path to users JSON
        orders_file: Path to orders CSV
        
    Returns:
        Dictionary with user preferences and recent purchases
    """
    users = load_users(users_file)
    orders = load_orders(orders_file)
    products = load_products()
    
    if user_id not in users:
        return None
    
    user = users[user_id]
    
    # Get user's orders
    user_orders = [o for o in orders if o['user_id'] == user_id]
    purchased_product_ids = [o['product_id'] for o in user_orders]

    # Identify most recent order if available
    last_order = None
    if user_orders:
        latest = max(user_orders, key=lambda o: o['timestamp'])
        last_product = get_product_by_id(products, latest['product_id'])
        if last_product:
            last_order = {
                'order_id': latest['order_id'],
                'product_id': latest['product_id'],
                'title': last_product['title'],
                'category': last_product['category'],
                'region': last_product['region'],
                'tags': last_product.get('tags', []),
                'description': last_product.get('description', '')
            }
    
    return {
        'user_id': user_id,
        'name': user['name'],
        'segment': user['segment'],
        'preferred_categories': user['preferred_categories'],
        'preferred_regions': user['preferred_regions'],
        'purchased_products': purchased_product_ids,
        'order_count': len(user_orders),
        'last_order': last_order
    }


if __name__ == "__main__":
    # Test the loader
    print("Loading products...")
    products = prepare_products_for_indexing()
    print(f"Loaded {len(products)} products")
    
    print("\nSample product:")
    sample = products[0]
    print(f"ID: {sample['id']}")
    print(f"Text preview: {sample['text'][:200]}...")
    print(f"Metadata: {sample['metadata']}")
    
    print("\nLoading user context...")
    user_context = get_user_context("USER-001")
    if user_context:
        print(f"User: {user_context['name']}")
        print(f"Segment: {user_context['segment']}")
        print(f"Preferred categories: {user_context['preferred_categories']}")
        print(f"Purchased products: {user_context['purchased_products']}")
