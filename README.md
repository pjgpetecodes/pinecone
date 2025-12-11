# Pinecone Recommender Systems - Personalized Product Search

A Python application demonstrating personalized product recommendations using Pinecone vector search with metadata filtering. This branch showcases how to build recommender systems that combine semantic search with user preferences and purchase history.

## What This Branch Demonstrates

This implementation shows:
- **Metadata-driven filtering**: Filter products by category, region, price range
- **User personalization**: Recommendations based on user preferences and segments
- **Popularity boosting**: Surface trending items alongside relevant matches
- **Purchase history awareness**: Highlight previously purchased items
- **Rich product metadata**: Categories, tags, regions, pricing, popularity scores

## Prerequisites

- Docker Desktop installed
- Python 3.8 or higher
- Azure OpenAI account with an embedding model deployed
- A PDF file to process

## Setup Instructions

### 1. Clone and Navigate to Project

```bash
cd pinecone1
```

### 2. Create Python Virtual Environment

In VS Code, use the Command Palette (Ctrl+Shift+P) and run:
- **Python: Create Environment**

Or from the command line:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

For text-based search:
```powershell
pip install pinecone python-dotenv openai
```

For multimodal search (text + images) using **offline CLIP**:
```powershell
pip install pinecone python-dotenv openai pillow requests torch transformers
```

Or install all at once:
```powershell
pip install pinecone python-dotenv openai pillow requests torch transformers
```

**Note:** `torch` and `transformers` are required for offline CLIP image embeddings. The first run will download the CLIP model (~350MB) and cache it locally.

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=pclocal
AZURE_OPENAI_KEY=<your-azure-openai-key>
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_INSTANCE_NAME=<your-instance-name>
AZURE_OPENAI_DEPLOYMENT_NAME=gpt4o
AZURE_OPENAI_EMBED_DEPLOYMENT_NAME=embeddingmodel
```

**Note:** Multimodal image search uses offline CLIP and does not require Azure OpenAI Vision API credentials.

### 5. Start Pinecone Local with Docker

Remove any existing container and start fresh:

```powershell
docker rm -f pinecone-local
docker run -d --name pinecone-local `
  -e PORT=5081 -e PINECONE_HOST=localhost `
  -p 5081:5081 -p 5082:5082 `
  --platform linux/amd64 `
  ghcr.io/pinecone-io/pinecone-local:latest
```

Wait a few seconds for the container to fully start:

```powershell
Start-Sleep -Seconds 3
docker logs pinecone-local
```

Verify both ports are exposed:

```powershell
docker ps --filter name=pinecone-local --format "table {{.Names}}\t{{.Ports}}"
```

You should see: `0.0.0.0:5081-5082->5081-5082/tcp`

## Store Data Overview

This demo includes a fake store with:

### Products (20 items)
- **Categories**: Electronics, Fitness, Kitchen, Home Office, Food & Beverage, Outdoor, Travel, Accessories
- **Regions**: North America, Europe, Asia
- **Metadata**: Price, popularity scores, tags, creation dates
- Located in: `data/store/products.json`

### Users (10 profiles)
- **Segments**: fitness-enthusiast, tech-professional, eco-conscious, wellness-seeker, home-chef, remote-worker, outdoor-adventurer, gadget-lover
- **Preferences**: Preferred categories and regions per user
- **Overlap users**: USER-009 and USER-010 added to demonstrate segment-based "also popular" patterns
- Located in: `data/store/users.json`

### Orders (59 transactions)
- User purchase history with customer ratings (1-5 stars)
- Used to calculate average product ratings
- Varied ratings create realistic differentiation (some products 4.5+, others 2-3 stars)
- Located in: `data/store/orders.csv`

## Running the Recommender System

### Step 1: Index Products

Create the vector index with product embeddings:

```powershell
python index_products.py
```

This will:
1. Load products from `data/store/products.json`
2. Load orders to calculate popularity scores
3. Generate embeddings for product descriptions
4. Create a Pinecone index with rich metadata
5. Upsert all product vectors

**Output:**
```
Loading products from store data...
Loaded 20 products

Generating embeddings for products...
Generated 20 embeddings

Upserted batch 1 (20 vectors)

Total product vectors upserted: 20

Index Statistics:
  Total vectors: 20
  Dimension: 1536

Product index ready for recommender queries!
```

### Step 2: Run the Guided Tour (Default)

Experience four real-world recommendation scenarios:

```powershell
python query_products.py
```

This runs the **guided tour** by default, demonstrating:

1. **Just purchased → You might also like**
   - User just bought something; suggest related items without re-showing the purchase
   - Uses: USER-002 (tech-professional), last purchase mode

2. **Welcome back → Because you bought before**
   - Logged-in user with history; use preferences and purchases to suggest next picks
   - Uses: USER-005 (home-chef), query: "kitchen essentials"

3. **Active search → Closest match + related**
   - User searches for something specific; combine similarity, preferences & ratings
   - Uses: USER-010 (tech-professional), query: "wireless headphones"

4. **Also popular in your crowd**
   - Show segment overlap where fitness enthusiasts discover shared popular purchases
   - Uses: USER-009 (fitness-enthusiast), query: "fitness gear"

After the tour completes, it automatically switches to interactive mode for free exploration.

### Step 3: Interactive Mode

Skip the tour and go straight to manual queries:

```powershell
python query_products.py --interactive
```

**Available Users:**
- `USER-001`: Alice (fitness-enthusiast) - prefers Fitness, Food & Beverage
- `USER-002`: Bob (tech-professional) - prefers Electronics, Home Office
- `USER-003`: Carol (eco-conscious) - prefers Kitchen, Outdoor
- `USER-004`: David (wellness-seeker) - prefers Fitness, Food & Beverage
- `USER-005`: Emma (home-chef) - prefers Kitchen, Food & Beverage
- `USER-006`: Frank (remote-worker) - prefers Home Office, Electronics
- `USER-007`: Grace (outdoor-adventurer) - prefers Outdoor, Travel, Fitness
- `USER-008`: Henry (gadget-lover) - prefers Electronics, Accessories
- `USER-009`: Isabella (fitness-enthusiast) - prefers Fitness, Outdoor
- `USER-010`: Jason (tech-professional) - prefers Electronics, Home Office

**Example Session:**
```
🔍 Enter search query (or press Enter to skip and use last purchase): workout equipment
👤 Enter User ID for personalization: USER-001
📁 Filter by category (or press Enter to skip): 
🌍 Filter by region (or press Enter to skip): 

🔍 Personalizing results for: Alice Johnson (fitness-enthusiast)
   Preferred categories: Fitness, Food & Beverage
   Preferred regions: North America
   Previous purchases: 4 orders

🎯 Applied filters: {'category': {'$in': ['Fitness', 'Food & Beverage']}, 'region': {'$in': ['North America']}}

🛍️  TOP RECOMMENDATIONS (5 results)

1. Adjustable Dumbbell Set
   Category: Fitness | Region: North America
   Price: $299.99 | Avg Rating: 4.33/5.0
   Score: 0.8976
   ✅ Previously purchased
   Tags: dumbbells, weights, home-gym, strength

2. Organic Protein Powder Vanilla
   Category: Food & Beverage | Region: North America
   Price: $44.99 | Avg Rating: 3.0/5.0
   Score: 0.8543
   Tags: protein, organic, plant-based, nutrition

3. Resistance Bands Exercise Set
   Category: Fitness | Region: Europe
   Price: $29.99 | Avg Rating: 4.0/5.0
   Score: 0.8321
   ✅ Previously purchased
   Tags: resistance, exercise, home-workout, strength
```

**Special feature:** Leave the query blank and provide a User ID to see recommendations based on their last purchase (pure similarity, no category filters).

### Step 3: Index Product Images (Multimodal Search - Offline CLIP)

Generate embeddings for product images using **offline CLIP** (no API calls needed):

```powershell
python index_images.py
```

This will:
1. Download and cache the CLIP model locally (~350MB, one-time)
2. Analyze each product image using CLIP (completely offline)
3. Generate 512-dimensional image embeddings (or 768 depending on CLIP variant)
4. Store image vectors alongside text vectors in Pinecone
5. Enable hybrid search combining text and image modalities

**Why CLIP?**
- ✅ Completely offline - no API calls to external services
- ✅ Free and open-source (from OpenAI)
- ✅ Understands both text and images in same embedding space
- ✅ Fast once model is cached
- ✅ No authentication required
- ✅ Same concept as Pinecone's JavaScript implementation with transformers.js

**Output:**
```
Initializing CLIP model for offline image embeddings...
✓ CLIP initialized. Embedding dimension: 512

Loading products...
Loaded 20 products

Generating image embeddings using CLIP (offline)...
Processing images - this may take a few minutes on first run (downloading model cache)...

  Processing PROD-001: Wireless Bluetooth Headphones...
  Processing PROD-002: Organic Cotton Yoga Mat...
  [... more products ...]

✓ Generated embeddings for 20 product images
✓ Upserted batch 1 (20 image vectors)

============================================================
Image indexing complete!
============================================================
✓ Total image vectors stored: 20
✓ Total index size: 40 vectors (text + image)
✓ Embedding dimension: 512
✓ All processing done offline using CLIP
```

### Step 4: Run Multimodal Search

Search using text, images, or both:

```powershell
python query_multimodal.py
```

**Search Options:**

1. **Text Search** - Traditional keyword-based search
   ```
   Query: "yoga mat for meditation"
   Results: Organic Cotton Yoga Mat (0.876), Meditation Cushion Zafu (0.843), ...
   ```

2. **Image Search** - Find similar products by image
   ```
   Image URL: https://images.unsplash.com/photo-1575311373937-040b8e1fd5b6?w=400
   Results: Smart Fitness Tracker Watch (0.912), Adjustable Dumbbell Set (0.856), ...
   ```

3. **Hybrid Search** - Combine text and image queries
   ```
   Text: "wireless audio"
   Image: https://images.unsplash.com/photo-1608043152269-423dbba4e7e1?w=400
   Results: [Combined rankings by text (60%) + image (40%) similarity]
   ```

4. **Personalized Search** - Text search with user preferences
   ```
   User: USER-001 (fitness-enthusiast)
   Query: "strength training equipment"
   Results: [Filtered by user's preferred categories and regions]
   ```

### Step 5: Run Legacy Example Queries

See older pre-configured searches:

```powershell
python query_products.py --examples
```

This runs demonstrations for:
- Tech professional searching for audio gear
- Fitness enthusiast looking for workout equipment
- Eco-conscious shopper finding sustainable kitchen items
- Remote worker setting up home office

## Key Features Demonstrated

### 1. Guided Tour Mode (Default - Text Search)
Run realistic recommendation scenarios that demonstrate the full system:

```powershell
python query_products.py  # Default: runs guided tour
```
- Four pre-configured scenarios showing different recommendation patterns
- Automatic progression with explanations
- Transitions to interactive mode after completion

### 2. Metadata Filtering
Filter products by multiple attributes simultaneously:
```python
# Category + Region filtering (active search mode)
metadata_filter = {
    'category': {'$in': ['Electronics', 'Home Office']},
    'region': {'$in': ['North America', 'Asia']}
}
```

### 3. User Personalization with Two Modes
**Active search mode** (user enters query):
```python
user_context = get_user_context("USER-002")  # Bob - tech professional
# Filters applied: categories=[Electronics, Home Office], regions=[Asia, North America]
```

**Last-purchase mode** (query blank):
```python
# Seed from user's last order, skip preference filters for pure similarity
query_vector = embed(last_purchase_description)
# NO category/region filters → finds truly similar items
```

### 4. Average Rating Display
Show customer satisfaction alongside similarity scores:
```python
avg_rating = calculate_average_ratings(products, orders)
# Products with high ratings (4.5+) vs. mixed ratings (2-3) stand out
```

### 5. Price Range Filtering
Add price constraints to queries:
```python
filters['price'] = {'$gte': 20.0, '$lte': 100.0}
```

### 6. Command-Line Modes
```powershell
python query_products.py              # Guided tour (default)
python query_products.py --interactive # Skip tour, manual queries
python query_products.py --examples    # Legacy pre-configured demos
```

## Architecture Overview

```
┌─────────────────┐
│  products.json  │──┐
│  users.json     │  │
│  orders.csv     │  │
└─────────────────┘  │
                     ▼
              ┌──────────────┐
              │ product_loader│
              │  - Load data  │
              │  - Calculate  │
              │    popularity │
              └──────┬────────┘
                     │
                     ▼
              ┌──────────────┐
              │   Embedding   │◄──── Azure OpenAI
              │    Helper     │      text-embedding-ada-002
              └──────┬────────┘
                     │
                     ▼
              ┌──────────────┐
              │   Pinecone   │
              │Product Index │
              │  + Metadata  │
              └──────┬────────┘
                     │
                     ▼
              ┌──────────────┐
              │query_products│
              │ - User prefs │
              │ - Filters    │
              │ - Boost      │
              └──────────────┘
```

## File Structure

```
.
├── data/
│   └── store/
│       ├── products.json      # 20 product catalog
│       ├── users.json         # 8 user profiles
│       └── orders.csv         # 35 purchase records
├── product_loader.py          # Data loading and preparation
├── index_products.py          # Index creation script
├── query_products.py          # Personalized query interface
├── embedding_helper.py        # Azure OpenAI wrapper
└── README.md                  # This file
```

## How the Recommender System Works

### 1. Vector Embeddings: Turning Products into Numbers

Each product is converted into a **vector** (a list of 1,536 numbers) using Azure OpenAI's embedding model. This vector captures the *semantic meaning* of the product based on its title, description, category, tags, price, and region.

**Example:**
```
"Meditation Cushion" → [0.02, -0.15, 0.38, ..., 0.11]  (1,536 numbers)
"Yoga Mat"          → [0.03, -0.14, 0.40, ..., 0.09]  (similar pattern!)
```

Products with similar purposes, materials, or use cases get similar vectors. This is how the system "understands" that a yoga mat and meditation cushion are related, even without shared keywords.

### 2. Vector Similarity Search: Finding "Close" Items

When you search or seed from a purchase, the system:

1. **Converts your input to a vector** using the same embedding model
2. **Uses Pinecone to find products** whose vectors are *closest* in 1,536-dimensional space
3. **Measures closeness with cosine similarity** (0 to 1, where 1 = identical)

**Result:** Items that are conceptually similar rise to the top, even if they don't share exact keywords.

```
Query: "wireless headphones"
Vector: [0.21, -0.08, 0.45, ...]
         ↓
Finds:  Bluetooth earbuds [0.22, -0.07, 0.44, ...] ✓ Score: 0.89
        Noise-canceling headphones [0.20, -0.09, 0.46, ...] ✓ Score: 0.87
```

### 3. Metadata Filtering: Narrowing by Facts

Before or during vector search, we filter by structured data:
- **Category** (Fitness, Electronics, Kitchen, etc.)
- **Region** (North America, Europe, Asia)
- **Price range** ($10-$50)
- **Average rating** (customer reviews)

**Two modes in this demo:**

#### Active Search Mode (user enters a query)
Filters by user preferences to personalize results:
```python
User: "wireless headphones" + preference: Electronics, Asia
      ↓
Filter: category IN [Electronics] AND region IN [Asia, North America]
      ↓
Result: Only electronics from preferred regions
```

#### Last-Purchase Mode (no query entered)
Skips preference filters to let vector similarity find truly similar items:
```python
User just bought: Meditation Cushion (no query entered)
      ↓
NO category/region filters applied
      ↓
Result: Yoga mats, resistance bands, wellness items across all regions
```

### 4. The Complete Flow

**Scenario 1: Active Search with Personalization**
```
User: Bob (tech-professional)
Query: "wireless headphones"
      ↓
1. Convert query to vector
      ↓
2. Apply filters: category IN [Electronics, Home Office]
                  region IN [Asia, North America]
      ↓
3. Pinecone finds 5 nearest vectors matching filters
      ↓
4. Results: Bluetooth headphones, USB-C hubs, desk lamps
   (All in preferred categories/regions, ranked by similarity)
```

**Scenario 2: Last-Purchase Recommendations**
```
User: Bob (tech-professional)
Last Purchase: Meditation Cushion Zafu
No query entered
      ↓
1. Convert cushion details to vector
      ↓
2. NO filters applied (pure similarity)
      ↓
3. Pinecone finds 5 nearest vectors
      ↓
4. Results: Yoga mats, resistance bands, protein powder
   (Semantically similar items, ignoring Bob's usual tech preferences)
```

### 5. Why This Architecture Matters

✅ **Semantic understanding**: Finds "workout gear" even if you search "exercise equipment"

✅ **Cross-category discovery**: "meditation cushion" suggests "yoga mat" (different products, similar intent)

✅ **Personalization without silos**: Filters respect preferences but don't trap users in echo chambers

✅ **Scalable filtering**: Pinecone filters at query time—no need to maintain separate indexes per user

✅ **Real-time adaptation**: Add new products or user preferences instantly without re-indexing

### 6. Under the Hood: Pinecone Metadata Schema

Each product vector in Pinecone stores:

```python
{
  "id": "PROD-002",
  "values": [0.03, -0.14, 0.40, ...],  # 1,536-dim vector
  "metadata": {
    "product_id": "PROD-002",
    "title": "Organic Cotton Yoga Mat",
    "category": "Fitness",           # ← Filterable
    "region": "North America",       # ← Filterable
    "price": 49.99,                  # ← Filterable (range)
    "avg_rating": 4.67,              # ← Display only
    "tags": ["yoga", "organic", ...], # ← Searchable
    "created_at": "2024-02-10"
  }
}
```

**Filtering happens at query time:**
```python
results = index.query(
    vector=query_vector,
    top_k=5,
    filter={
        "category": {"$in": ["Fitness", "Food & Beverage"]},
        "region": {"$in": ["North America", "Europe"]},
        "price": {"$lte": 100.0}
    }
)
```

Pinecone efficiently finds the top-5 most similar vectors **that also match all filters**.

## Concepts Illustrated

### Multimodal Search (Image + Text) - Offline CLIP
This branch extends the recommender system with multimodal capabilities using **offline CLIP**:
- **Text search**: Query products by description, category, or features
- **Image search**: Upload or reference an image to find similar products
- **Hybrid search**: Combine text and image queries for richer results
- **Offline processing**: No API calls - uses local CLIP model cached on disk
- **Dual embeddings**: Store both text (1536-dim) and image (512-dim) vectors in Pinecone
- **Weighted ranking**: Adjust importance between text and image matches

**How it works (completely offline):**
1. **CLIP Model**: OpenAI's CLIP (Contrastive Language-Image Pre-training) understands both text and images
2. **Image Processing**: Download image → Process with CLIP → Generate 512-dimensional embedding
3. **Local Execution**: Everything runs on your machine - no external API calls
4. **Unified Space**: Text and image embeddings are semantically comparable (CLIP's superpower!)
5. **Hybrid Queries**: Search by text, image, or both, with adjustable weighting

**Advantages over cloud-based approaches:**
- ✅ No API keys or authentication needed (except Pinecone)
- ✅ No rate limiting or quota concerns
- ✅ No streaming costs for image analysis
- ✅ All processing stays on your machine (privacy)
- ✅ Model cached after first download (~350MB)
- ✅ Compatible with transformers.js (Pinecone's JavaScript approach)

**Schema with Offline Multimodal Support:**
```json
Text Vector (PROD-001):
  id: "PROD-001"
  values: [0.123, -0.456, 0.789, ...]  // 1536 dimensions (from text embedding)
  metadata: {
    product_id: "PROD-001",
    title: "Wireless Bluetooth Headphones",
    image_url: "https://unsplash.com/...",
    vector_type: "text"
  }

Image Vector (PROD-001-image):
  id: "PROD-001-image"
  values: [0.234, -0.567, 0.890, ...]  // 512 dimensions (from CLIP image embedding)
  metadata: {
    product_id: "PROD-001",
    title: "Wireless Bluetooth Headphones",
    image_url: "https://unsplash.com/...",
    vector_type: "image"
  }
```

### Vector Search + Metadata Filtering
Pinecone allows combining semantic search with structured filters:
- Vector similarity finds semantically related products
- Metadata filters constrain results to user preferences
- Results satisfy both meaning AND constraints

### Personalization Strategies
1. **Preference-based**: Filter by user's favorite categories/regions (active search only)
2. **History-aware**: Highlight previously purchased items, exclude from "you might also like"
3. **Collaborative signals**: Calculate average ratings from customer reviews (1-5 stars)
4. **Hybrid modes**: 
   - Active search (with preference filters) 
   - Last-purchase (pure similarity, no filters)
5. **Segment overlap**: Users in same segment share purchase patterns for "also popular" recommendations

### Metadata Schema Design
Product vectors include:
- **Searchable fields**: product_id, title, category, tags
- **Filterable fields**: category, region, price range
- **Ranking signals**: avg_rating (derived from order ratings)
- **Display data**: price, creation date, full metadata

## Troubleshooting

### Container Not Running
```powershell
docker ps -a
docker start pinecone-local
```

### Index Not Found
Re-run `index_products.py` to create the product index.

### No Results
Check filters - too many constraints may exclude all products. Try:
- Removing user_id to search without personalization
- Skipping category/region filters
- Widening price range

### Connection Errors
Ensure `.env` has correct Azure OpenAI credentials and the Pinecone container is running on ports 5081-5082.

## Learning Resources

This branch demonstrates concepts from the Pluralsight course:
- **Learning Objective**: Develop recommender systems using Pinecone metadata filtering and vector similarity to surface personalized results
- **Key Skills**: Metadata schema design, user preference modeling, hybrid ranking, popularity boosting

Explore other branches:
- `main` - Basic RAG with PDF documents
- `3-multimodal-search` - Image + text hybrid search
- `4-anomaly-detection` - Log analysis with vector outliers
- `5-index-design` - Schema best practices
- `6-performance-evaluation` - Metrics and optimization

---

**Next Steps**: Experiment with different user profiles, add new products, or modify the popularity calculation to see how recommendations change!
