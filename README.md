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

```powershell
pip install pinecone python-dotenv openai
```

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

### Users (8 profiles)
- **Segments**: fitness-enthusiast, tech-professional, eco-conscious, wellness-seeker, home-chef, remote-worker, outdoor-adventurer, gadget-lover
- **Preferences**: Preferred categories and regions per user
- Located in: `data/store/users.json`

### Orders (35 transactions)
- User purchase history with ratings
- Used to calculate dynamic popularity scores
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

### Step 2: Query with Personalization

Run the interactive query interface:

```powershell
python query_products.py
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

**Example Session:**
```
🔍 Enter search query: workout equipment
👤 Enter User ID: USER-001
📁 Filter by category: (press Enter)
🌍 Filter by region: (press Enter)

🔍 Personalizing results for: Alice Johnson (fitness-enthusiast)
   Preferred categories: Fitness, Food & Beverage
   Preferred regions: North America
   Previous purchases: 4 orders

🎯 Applied filters: {'category': {'$in': ['Fitness', 'Food & Beverage']}, 'region': {'$in': ['North America']}}

🛍️  TOP RECOMMENDATIONS (5 results)

1. Adjustable Dumbbell Set
   Category: Fitness | Region: North America
   Price: $299.99 | Popularity: 9.3/10.0
   Score: 0.8976
   ✅ Previously purchased
   Tags: dumbbells, weights, home-gym, strength

2. Organic Protein Powder Vanilla
   Category: Food & Beverage | Region: North America
   Price: $44.99 | Popularity: 7.6/10.0
   Score: 0.8543
   ✅ Previously purchased
   Tags: protein, organic, plant-based, nutrition

3. Resistance Bands Exercise Set
   Category: Fitness | Region: North America (via Europe fallback)
   Price: $29.99 | Popularity: 8.0/10.0
   Score: 0.8321
   Tags: resistance, exercise, home-workout, strength
```

### Step 3: Run Example Queries

See pre-configured personalized searches:

```powershell
python query_products.py --examples
```

This runs demonstrations for:
- Tech professional searching for audio gear
- Fitness enthusiast looking for workout equipment
- Eco-conscious shopper finding sustainable kitchen items
- Remote worker setting up home office

## Key Features Demonstrated

### 1. Metadata Filtering
Filter products by multiple attributes simultaneously:
```python
# Category + Region filtering
metadata_filter = {
    'category': {'$in': ['Electronics', 'Home Office']},
    'region': {'$in': ['North America', 'Asia']}
}
```

### 2. User Personalization
Automatically apply user preferences:
```python
user_context = get_user_context("USER-002")  # Bob - tech professional
# Filters applied: categories=[Electronics, Home Office], regions=[Asia, North America]
```

### 3. Popularity Boosting
Combine semantic similarity with popularity scores:
```python
popularity_factor = 1 + (popularity / 20)  # 1.0 to 1.5x boost
boosted_score = similarity_score * popularity_factor
```

### 4. Price Range Filtering
Add price constraints to queries:
```python
filters['price'] = {'$gte': 20.0, '$lte': 100.0}
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

## Concepts Illustrated

### Vector Search + Metadata Filtering
Pinecone allows combining semantic search with structured filters:
- Vector similarity finds semantically related products
- Metadata filters constrain results to user preferences
- Results satisfy both meaning AND constraints

### Personalization Strategies
1. **Preference-based**: Filter by user's favorite categories/regions
2. **History-aware**: Highlight previously purchased items
3. **Collaborative signals**: Use order counts to boost popular items
4. **Hybrid scoring**: Blend vector similarity with popularity

### Metadata Schema Design
Product vectors include:
- **Searchable fields**: product_id, title, category, tags
- **Filterable fields**: category, region, price range
- **Ranking signals**: popularity score (derived from orders)
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
