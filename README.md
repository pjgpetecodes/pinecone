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
1. Download and cache the CLIP model locally (~350MB, one-time only)
2. Download and cache each product image to `data/store/images/` (locally stored, never re-downloaded)
3. Analyze each product image using CLIP (completely offline)
4. Generate 512-dimensional image embeddings and pad to 1536-dim to match text vectors
5. Store image vectors alongside text vectors in Pinecone
6. Enable hybrid search combining text and image modalities

**Why CLIP?**
- ✅ Completely offline - no API calls to external services
- ✅ Free and open-source (from OpenAI)
- ✅ Understands both text and images in same embedding space
- ✅ Fast once model and images are cached
- ✅ No authentication required
- ✅ Same concept as Pinecone's JavaScript implementation with transformers.js
- ✅ Local image caching ensures reliability even if external URLs become unavailable

**First Run (downloads CLIP model and images):**
```
Loading CLIP model: openai/clip-vit-base-patch32
Using device: cuda
CLIP model loaded. Native dimension: 512, Target dimension: 1536
Image cache directory: data/store/images

Loading products...
Loaded 20 products

Generating image embeddings using CLIP (offline)...
Processing images - this may take a few minutes on first run (downloading model cache)...

  Processing 1/20: Wireless Bluetooth Headphones...
  Processing 2/20: Organic Cotton Yoga Mat...
  [... downloading and caching images ...]
  Processing 20/20: Wireless Charging Pad Fast Charge...

✓ Generated embeddings for 20 product images
✓ Upserted batch 1 (20 image vectors)

============================================================
Image indexing complete!
============================================================
✓ Total image vectors stored: 20
✓ Total index size: 40 vectors (text + image)
✓ CLIP native dimension: 512D (padded to 1536D)
✓ All processing done offline using CLIP
```

**Subsequent Runs (loads from cache, much faster):**
```
[CLIP model loads from cache in seconds]
[Images load from data/store/images/ directory, no downloads needed]
✓ Generated embeddings for 20 product images
✓ All processing complete in ~30 seconds
```

**Image Caching Benefits:**
- 📁 All images stored in `data/store/images/` with names like `PROD-001.jpg`, `PROD-002.jpg`, etc.
- ⚡ Subsequent indexing runs are 10x faster (no network downloads)
- 🔒 Reliable - doesn't depend on external URLs remaining available
- 🎓 Great for course demos and reproducible results

### Step 4: Run Multimodal Search

Search using text, images, or both:

```powershell
python query_multimodal.py
```

**Interactive Menu Options:**

1. **Text Search** - Traditional keyword-based search
   ```
   Search Option: 1
   Enter your search query: yoga mat for meditation
   
   Results: 
   1. Organic Cotton Yoga Mat
      Category: Fitness | Price: $49.99 | Rating: 4.67/5.0
      Similarity Score: 0.876
      
   2. Meditation Cushion Zafu
      Category: Accessories | Price: $89.99 | Rating: 4.0/5.0
      Similarity Score: 0.843
   ```

2. **Image Search** - Find similar products by image (local file or URL)
   ```
   Search Option: 2
   Enter image path or URL to search with: data/store/images/PROD-003.jpg
   
   Analyzing image and searching...
   
   Results:
   1. Smart Fitness Tracker Watch
      Category: Electronics | Price: $199.99 | Rating: 4.5/5.0
      Similarity Score: 0.912
      
   2. Adjustable Dumbbell Set
      Category: Fitness | Price: $299.99 | Rating: 4.33/5.0
      Similarity Score: 0.856
   ```

3. **Hybrid Search** - Combine text and image queries (60% text, 40% image)
   ```
   Search Option: 3
   Enter text query: wireless audio equipment
   Enter image path or URL (or press Enter to skip): data/store/images/PROD-001.jpg
   
   Processing hybrid search...
   
   Results (combined text + image):
   1. Wireless Bluetooth Headphones
      Category: Electronics | Price: $149.99 | Rating: 4.8/5.0
      Combined Score: 0.923
   ```

4. **Personalized Search** - Text search with user preferences
   ```
   Search Option: 4
   Enter user ID (e.g., USER-001): USER-001
   Enter your search query: strength training equipment
   
   Results filtered by USER-001 (Alice, fitness-enthusiast):
   - Preferred categories: Fitness, Food & Beverage
   - Preferred regions: North America
   
   1. Adjustable Dumbbell Set
      Category: Fitness | Price: $299.99 | Rating: 4.33/5.0
   ```

**Example Workflows:**

**Workflow 1: Search by local product image**
```powershell
python query_multimodal.py

Search by Image (Option 2)
Path: data/store/images/PROD-003.jpg
↓
Finds: Similar fitness products
```

**Workflow 2: Hybrid text + image search**
```powershell
python query_multimodal.py

Hybrid Search (Option 3)
Text: "home office setup"
Image: data/store/images/PROD-008.jpg
↓
Results weighted: 60% text relevance + 40% visual similarity
```

**Workflow 3: Search any image (not just products)**
```powershell
python query_multimodal.py

Image Search (Option 2)
Path: C:\Users\YourName\Pictures\my-yoga-mat.jpg
↓
Finds: Products visually similar to your image
```

### Step 5: Understanding the Multimodal Index

After running `index_images.py`, your Pinecone index contains:

**Text Vectors (from product descriptions):**
```
ID: PROD-001
Vector: [0.123, -0.456, 0.789, ...]  (1536 dimensions from Azure OpenAI)
Type: Text embedding of product description
```

**Image Vectors (from product images):**
```
ID: PROD-001-image
Vector: [0.456, -0.123, 0.234, ...]  (512 dims from CLIP, padded to 1536)
Type: Image embedding from CLIP
```

**Total Index:**
- 20 text vectors (1 per product)
- 20 image vectors (1 per product)
- **40 total vectors** enabling rich multimodal search

**How Hybrid Ranking Works:**
1. Search text vectors for text query → get text-based rankings
2. Search image vectors for image input → get image-based rankings
3. Combine scores: `combined_score = (text_score × 0.6) + (image_score × 0.4)`
4. Return top-k by combined score

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
│       ├── products.json       # 20 product catalog with image_url field
│       ├── users.json          # 10 user profiles with segments
│       ├── orders.csv          # 59 purchase records for ratings
│       └── images/             # Cached product images (downloaded on first run)
│           ├── PROD-001.jpg
│           ├── PROD-002.jpg
│           └── ... (20 total)
├── product_loader.py           # Data loading and preparation
├── embedding_helper.py         # Azure OpenAI text embedding wrapper
├── image_helper.py             # CLIP-based image embedding and caching
├── index_products.py           # Create text vector index
├── index_images.py             # Create image vector index with caching
├── query_products.py           # Personalized text search interface
├── query_multimodal.py         # Multimodal search interface (text/image/hybrid)
├── pdf_helper.py               # PDF processing utilities
├── pinecone_indexes.py         # Index management utilities
└── README.md                   # This file
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

This implementation extends the recommender system with **multimodal capabilities using offline CLIP**:

**Four Search Modes:**
1. **Text-only search**: Query products by description, category, or features
2. **Image-only search**: Upload or reference an image to find similar products
3. **Hybrid search**: Combine text and image queries with adjustable weighting (default 60% text, 40% image)
4. **Personalized search**: Text search filtered by user preferences and segments

**How It Works (Completely Offline):**

1. **CLIP Model**: OpenAI's CLIP (Contrastive Language-Image Pre-training) understands both text and images in a shared embedding space
2. **Image Download & Cache**: Images are downloaded once and stored locally in `data/store/images/` for subsequent runs
3. **Embedding Generation**: 
   - Text: Azure OpenAI generates 1536-dimensional vectors
   - Images: CLIP generates 512-dimensional vectors (padded to 1536-dim for Pinecone compatibility)
4. **Unified Space**: Both text and image embeddings are semantically comparable in the same Pinecone index
5. **Hybrid Ranking**: 
   - Run both text and image searches
   - Normalize scores (1st place = 1.0, 2nd = 0.8, etc.)
   - Combine with weights: `score = (text_score × text_weight) + (image_score × (1-text_weight))`

**Architecture:**
```
Products (20)
    ↓
    ├─→ Text embedding (Azure OpenAI) ──→ PROD-001 (text vector, 1536-dim)
    │                                         ↓
    │                                    Pinecone Index
    │                                         ↑
    └─→ Image download & embedding ────→ PROD-001-image (image vector, 512-dim→1536-dim)
           (CLIP locally)
           
    User Query
         ↓
    ├─→ Text query → Search text vectors
    │                    ↓
    │             Rank results
    │
    ├─→ Image query → Search image vectors
    │                    ↓
    │             Rank results
    │
    └─→ Both → Combine rankings with weights → Final results
```

**Key Features:**

✅ **Completely Offline**
- No API calls to external services
- CLIP model cached locally (~350MB, downloaded once)
- Images cached locally, never re-downloaded
- Only Pinecone communication required

✅ **Zero Additional Costs**
- CLIP is free and open-source (from OpenAI)
- No per-image processing fees
- Single Pinecone index (no separate image index needed)

✅ **Fast Iterations**
- First run: Download CLIP model + cache images (5-10 minutes)
- Subsequent runs: Load from cache instantly (~30 seconds for indexing)

✅ **Reliable & Reproducible**
- Cached images in `data/store/images/` ensure consistent results
- No dependency on external URL availability
- Perfect for course demos and testing

✅ **Flexible Input**
- Image search accepts local file paths or URLs
- Hybrid search combines any text query with any image
- Personalized search applies user preferences to any query mode

**Advantages over Cloud-Based Approaches:**
- ✅ No API keys or authentication for image processing
- ✅ No rate limiting or quota concerns
- ✅ No streaming costs for image analysis
- ✅ All processing stays on your machine (privacy)
- ✅ Model cached after first download
- ✅ Compatible with Pinecone's transformers.js approach (JavaScript)

**Multimodal Index Schema:**
```json
Text Vector (PROD-001):
{
  "id": "PROD-001",
  "values": [0.123, -0.456, 0.789, ...],  // 1536 dims from Azure OpenAI
  "metadata": {
    "product_id": "PROD-001",
    "title": "Wireless Bluetooth Headphones",
    "category": "Electronics",
    "image_url": "https://images.unsplash.com/...",
    "vector_type": "text"
  }
}

Image Vector (PROD-001-image):
{
  "id": "PROD-001-image",
  "values": [0.234, -0.567, 0.890, ...],  // 512 dims from CLIP, padded to 1536
  "metadata": {
    "product_id": "PROD-001",
    "title": "Wireless Bluetooth Headphones",
    "category": "Electronics",
    "image_url": "https://images.unsplash.com/...",
    "vector_type": "image"
  }
}
```

**Example Hybrid Search Calculation:**
```python
# Both queries run against same Pinecone index
text_results = index.query(text_vector, top_k=5)
image_results = index.query(image_vector, top_k=5)

# Normalize results to scores (position-based)
# 1st = 1.0, 2nd = 0.8, 3rd = 0.6, 4th = 0.4, 5th = 0.2
text_scores = {prod_id: 1.0 - (i/5) for i, prod_id in enumerate(text_results)}
image_scores = {prod_id: 1.0 - (i/5) for i, prod_id in enumerate(image_results)}

# Combine with default weighting (60% text, 40% image)
for product_id in all_products:
    combined = (text_scores.get(product_id, 0) * 0.6) + 
               (image_scores.get(product_id, 0) * 0.4)
    
# Return top-5 by combined score
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
