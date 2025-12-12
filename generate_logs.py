import random
import uuid
from datetime import datetime, timedelta
from typing import List

from log_event_schema import LogEvent

# Set seed for reproducible demo logs
random.seed(42)


LEVELS = ["INFO", "WARN", "ERROR"]
SERVICES = ["auth", "payments", "cart", "orders"]

NORMAL_COUNTRIES = ["US", "CA", "UK", "DE", "FR", "AU"]
ANOMALY_COUNTRIES = ["RU", "CN", "NG", "RO", "VN"]

# User profiles with typical purchase categories
USER_PROFILES = {
    "user_0001": {"categories": ["fitness", "nutrition", "sportswear"], "name": "fitness enthusiast"},
    "user_0002": {"categories": ["electronics", "gaming", "tech"], "name": "tech enthusiast"},
    "user_0003": {"categories": ["books", "office", "education"], "name": "academic"},
    "user_0004": {"categories": ["home", "kitchen", "appliances"], "name": "homemaker"},
    "user_0005": {"categories": ["fashion", "beauty", "accessories"], "name": "fashion lover"},
    "user_0006": {"categories": ["outdoor", "camping", "hiking"], "name": "outdoor adventurer"},
    "user_0007": {"categories": ["baby", "kids", "toys"], "name": "parent"},
    "user_0008": {"categories": ["automotive", "tools", "hardware"], "name": "DIY enthusiast"},
    "user_0009": {"categories": ["music", "instruments", "audio"], "name": "musician"},
    "user_0010": {"categories": ["pet", "pet-food", "pet-toys"], "name": "pet owner"},
}

PRODUCT_CATEGORIES = {
    "fitness": ["yoga mat", "dumbbells", "resistance bands", "protein powder"],
    "nutrition": ["vitamins", "protein bars", "organic snacks"],
    "sportswear": ["running shoes", "gym shorts", "sports bra"],
    "electronics": ["laptop", "smartphone", "tablet", "smartwatch"],
    "gaming": ["gaming console", "controller", "gaming headset"],
    "tech": ["USB cable", "external drive", "webcam"],
    "books": ["textbook", "novel", "reference book"],
    "office": ["desk lamp", "notebook", "pen set"],
    "education": ["online course", "study guide"],
    "home": ["bedding", "curtains", "decorative pillow"],
    "kitchen": ["blender", "cookware set", "knife set"],
    "appliances": ["coffee maker", "toaster", "vacuum cleaner"],
    "fashion": ["designer dress", "handbag", "sunglasses"],
    "beauty": ["skincare set", "makeup palette", "perfume"],
    "accessories": ["jewelry", "scarf", "watch"],
    "outdoor": ["tent", "sleeping bag", "backpack"],
    "camping": ["camping stove", "lantern", "cooler"],
    "hiking": ["hiking boots", "trekking poles", "water bottle"],
    "baby": ["diapers", "baby monitor", "stroller"],
    "kids": ["kids clothing", "educational toy", "art supplies"],
    "toys": ["action figure", "board game", "puzzle"],
    "automotive": ["car battery", "motor oil", "tire gauge"],
    "tools": ["power drill", "wrench set", "screwdriver set"],
    "hardware": ["paint", "lumber", "nails"],
    "music": ["guitar strings", "sheet music", "microphone"],
    "instruments": ["keyboard", "guitar", "drumsticks"],
    "audio": ["studio headphones", "audio interface", "speakers"],
    "pet": ["dog bed", "cat tree", "pet carrier"],
    "pet-food": ["dog food", "cat food", "treats"],
    "pet-toys": ["chew toy", "ball", "scratching post"],
    # Unusual categories for anomalies
    "weapons": ["tactical knife", "survival gear", "pepper spray"],
    "gambling": ["poker chips", "casino dice", "betting guide"],
    "adult": ["adult novelty item", "adult book"],
}

USERS = list(USER_PROFILES.keys())

def _normal_event(user_id: str) -> dict:
    """Generate a normal event pattern for a specific user"""
    profile = USER_PROFILES[user_id]
    user_categories = profile["categories"]
    
    event_type = random.choice(["auth", "cart", "order"])
    
    if event_type == "auth":
        return {
            "level": "INFO",
            "service": "auth",
            "message": "user login successful",
            "amount": None,
            "country": random.choice(NORMAL_COUNTRIES),
            "category": None,
            "product": None,
        }
    elif event_type == "cart":
        category = random.choice(user_categories)
        product = random.choice(PRODUCT_CATEGORIES[category])
        return {
            "level": "INFO",
            "service": "cart",
            "message": f"item added to cart",
            "amount": round(random.uniform(10, 100), 2),
            "country": random.choice(NORMAL_COUNTRIES),
            "category": category,
            "product": product,
        }
    else:  # order
        category = random.choice(user_categories)
        product = random.choice(PRODUCT_CATEGORIES[category])
        return {
            "level": "INFO",
            "service": "orders",
            "message": f"order placed",
            "amount": round(random.uniform(50, 300), 2),
            "country": random.choice(NORMAL_COUNTRIES),
            "category": category,
            "product": product,
        }
    return random.choice(patterns)


def generate_synthetic_logs(n: int = 200, anomaly_ratio: float = 0.1) -> List[LogEvent]:
    now = datetime.utcnow()
    events: List[LogEvent] = []
    anomaly_count = int(n * anomaly_ratio)
    normal_count = n - anomaly_count

    # Normal events
    for i in range(normal_count):
        user = random.choice(USERS)
        pattern = _normal_event(user)
        ts = now - timedelta(seconds=random.randint(0, 7200))
        
        attrs = {"country": pattern["country"]}
        if pattern["amount"]:
            attrs["amount"] = pattern["amount"]
        if pattern["category"]:
            attrs["category"] = pattern["category"]
        if pattern["product"]:
            attrs["product"] = pattern["product"]
        
        events.append(
            LogEvent(
                timestamp=ts,
                level=pattern["level"],
                service=pattern["service"],
                message=pattern["message"],
                trace_id=str(uuid.uuid4()),
                user_id=user,
                attributes=attrs,
            )
        )

    # Specific, prescriptive anomalies for demo consistency
    predefined_anomalies = [
        # Anomaly 1: Fitness enthusiast buys weapons
        LogEvent(
            timestamp=now - timedelta(seconds=3600),
            level="WARN",
            service="orders",
            message="out-of-pattern purchase detected",
            trace_id="anomaly-001-category-mismatch",
            user_id="user_0001",  # fitness enthusiast
            attributes={
                "amount": 149.99,
                "country": "US",
                "category": "weapons",
                "product": "tactical knife",
                "user_profile": "fitness enthusiast",
                "typical_categories": "fitness, nutrition, sportswear",
            },
        ),
        # Anomaly 2: Tech enthusiast high-value purchase
        LogEvent(
            timestamp=now - timedelta(seconds=2800),
            level="WARN",
            service="payments",
            message="high value transaction flagged for review",
            trace_id="anomaly-002-high-value",
            user_id="user_0002",  # tech enthusiast
            attributes={
                "amount": 12499.00,
                "country": "US",
                "category": "electronics",
                "product": "laptop",
            },
        ),
        # Anomaly 3: Academic user login from Russia
        LogEvent(
            timestamp=now - timedelta(seconds=2400),
            level="WARN",
            service="auth",
            message="login from high-risk location detected",
            trace_id="anomaly-003-foreign-login",
            user_id="user_0003",  # academic
            attributes={"country": "RU"},
        ),
        # Anomaly 4: Parent buys gambling items
        LogEvent(
            timestamp=now - timedelta(seconds=1900),
            level="WARN",
            service="orders",
            message="out-of-pattern purchase detected",
            trace_id="anomaly-004-category-mismatch",
            user_id="user_0007",  # parent
            attributes={
                "amount": 89.99,
                "country": "CA",
                "category": "gambling",
                "product": "poker chips",
                "user_profile": "parent",
                "typical_categories": "baby, kids, toys",
            },
        ),
        # Anomaly 5: Pet owner login from China
        LogEvent(
            timestamp=now - timedelta(seconds=1500),
            level="WARN",
            service="auth",
            message="login from high-risk location detected",
            trace_id="anomaly-005-foreign-login",
            user_id="user_0010",  # pet owner
            attributes={"country": "CN"},
        ),
        # Anomaly 6: Homemaker high-value appliance purchase
        LogEvent(
            timestamp=now - timedelta(seconds=1200),
            level="WARN",
            service="payments",
            message="high value transaction flagged for review",
            trace_id="anomaly-006-high-value",
            user_id="user_0004",  # homemaker
            attributes={
                "amount": 8750.00,
                "country": "UK",
                "category": "appliances",
                "product": "vacuum cleaner",
            },
        ),
    ]
    
    events.extend(predefined_anomalies)

    random.shuffle(events)
    return events


if __name__ == "__main__":
    logs = generate_synthetic_logs(50, anomaly_ratio=0.2)
    for e in logs[:10]:
        print(e.to_text())
