# Pinecone Vector Search - Facial Similarity

**Branch**: `4-facial-similarity-search`  
**Objective**: Measure and rank facial similarity using Pinecone's vector search to compare embedding distances

## ⚠️ Important: Educational Purpose

**This is an EDUCATIONAL DEMONSTRATION of vector similarity measurement.**

This implementation:
- ✅ Teaches embedding generation and distance calculation concepts
- ✅ Shows how Pinecone enables similarity search on any embeddings
- ✅ Demonstrates cosine similarity in 128-dimensional space
- ✅ Uses public, diverse profile images (Unsplash stock photos)
- ✅ **Completely standalone** - not integrated with recommendations

This is **NOT** intended for:
- ❌ Real-world user profiling or identification
- ❌ Prediction or discrimination based on facial features
- ❌ Any non-educational use case

---

This branch demonstrates **facial similarity measurement** using:
- **Offline facial embeddings** using DeepFace Facenet model (no API calls, no costs)
- **Pinecone vector similarity search** to rank users by facial distance
- **Cosine similarity** for comparing high-dimensional face vectors
- **Educational context** - Learning vector embedding fundamentals

Built on top of the `3-multimodal-search` branch, this implementation focuses on facial feature embeddings while maintaining the same vector search principles used for products and images.

## What This Branch Adds

**Facial Similarity Capabilities:**
- ✅ Extract facial embeddings using offline DeepFace (Facenet model, 128-dim)
- ✅ Search for similar users by facial features via Pinecone
- ✅ Rank users by facial similarity scores (cosine distance)
- ✅ Separate index architecture (user-faces-index, 128-dim)
- ✅ Interactive query interface with multiple search modes

**Key Innovation**: Uses **offline DeepFace** for specialized facial recognition:
- No facial recognition API needed
- No per-image processing costs
- Industry-standard facial recognition model (99%+ accuracy on benchmarks)
- Privacy-preserving (all processing on your machine)
- Model cached locally (~350MB, downloaded once)

**Image Normalization & Consistency**:
- ✅ Automatic image resizing to 256×256 (aspect-preserving center-crop)
- ✅ Consistent preprocessing for both indexing and querying
- ✅ Improved embedding consistency across different image sizes and sources
- ✅ URL fallback to cached files when available for maximum consistency

## Prerequisites

- Docker Desktop installed (for Pinecone Local)
- Python 3.8 or higher
- No API keys required for facial embeddings (DeepFace runs offline)
- Azure OpenAI account only needed if integrating with other branches

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

For facial similarity search using **offline DeepFace**:
```powershell
pip install pinecone python-dotenv deepface opencv-python-headless pillow requests numpy
```

**Note:** `deepface` and `opencv-python-headless` are required for offline facial embeddings. The first run will download the Facenet model (~350MB) and cache it locally.

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=pclocal
```

**Note:** Facial similarity search uses offline DeepFace and does not require Azure OpenAI or any facial recognition API credentials.

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

## Quick Start - Facial Similarity Demo

### Step 1: Index User Facial Embeddings

```powershell
python index_user_faces.py
```

This will:
1. Load 10 user profiles with profile images
2. Extract facial embeddings using DeepFace (offline)
3. Cache images locally in `data/store/user_images/`
4. Store 128-dim embeddings in Pinecone `user-faces-index`

**First run**: Downloads DeepFace models (~350MB), takes 2-3 minutes  
**Subsequent runs**: Uses cached models, takes ~30 seconds

### Step 2: Run Facial Similarity Search

```powershell
python query_user_similarity.py
```

Interactive search interface with 4 options:
1. **Find similar by user ID** - Search for users with similar facial features
2. **Find similar by image** - Upload your own image to find similar users
3. **View all users** - List all indexed users
4. **Exit**

**Example Searches:**

```
Option 1: Find similar by user ID
Enter User ID: USER-001
Results:
  1. USER-004 (David Kim) - Similarity: 0.8532
  2. USER-006 (Frank Anderson) - Similarity: 0.8124
  3. USER-008 (Henry Brown) - Similarity: 0.7998
  ...

Option 2: Find similar by image
Enter image path: my_photo.jpg
Results:
  1. USER-003 (Carol Martinez) - Similarity: 0.8765
  2. USER-007 (Grace Lee) - Similarity: 0.8234
  ...
```

## Data Overview

### Users (10 profiles with facial images)
- **Segments**: fitness-enthusiast, tech-professional, eco-conscious, wellness-seeker, home-chef, remote-worker, outdoor-adventurer, gadget-lover
- **Profile Images**: Unsplash stock photos (diverse, public domain)
- **Embeddings**: 128-dimensional facial feature vectors via DeepFace Facenet
- Located in: `data/store/users.json`

### Facial Embeddings
- **Model**: DeepFace Facenet (industry-standard facial recognition)
- **Dimensions**: 128-dim vectors capturing facial features
- **Similarity Metric**: Cosine similarity (0-1 scale, 1=identical)
- **Index**: Separate Pinecone index (`user-faces-index`)

## How It Works

### 1. Facial Embedding Extraction (DeepFace)

```python
from face_embedding_helper import FaceEmbeddingHelper

helper = FaceEmbeddingHelper(model_name="Facenet")
embedding = helper.get_face_embedding(image_url, user_id)
# Returns 128-dim vector: [0.234, -0.567, 0.891, ...]
```

DeepFace analyzes facial features:
- Face detection and alignment
- Feature extraction (eyes, nose, mouth, face shape)
- Conversion to 128-dimensional vector representation
- Facenet model (99%+ accuracy on facial recognition benchmarks)

**Preprocessing Pipeline:**
- All images normalized to 256×256 (aspect-preserving center-crop)
- Applied during indexing and querying for consistency
- Ensures face detection alignment is identical across different image sizes
- Improves embedding consistency and match accuracy

### 2. Vector Storage (Pinecone)

```python
index.upsert(vectors=[{
    "id": "USER-001-face",
    "values": embedding,  # 128-dim vector
    "metadata": {
        "user_id": "USER-001",
        "name": "Alice Johnson",
        "profile_image_url": "https://...",
        "embedding_type": "facial"
    }
}])
```

Separate index architecture:
- **Index name**: `user-faces-index`
- **Dimensions**: 128 (Facenet output)
- **Metric**: Cosine similarity
- **Separate from products** - clean architecture

### 3. Similarity Search (Cosine Distance)

```python
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)
```

Pinecone uses cosine similarity:
- Measures angle between 128-dim vectors
- Scale: 0 (completely different) to 1 (identical)
- Returns ranked list of most similar faces
- Queries use same 256×256 preprocessing as indexed embeddings

### 4. Interactive Querying

Search by user ID or upload your own image for comparison:

```powershell
python query_user_similarity.py
````

# Option 1: Search by existing user
Enter User ID: USER-001
→ Returns users with similar facial features

# Option 2: Search by your own image
Enter image path: my_photo.jpg
→ Extracts embedding, finds similar users
```

## File Structure

```
pinecone/
├── face_embedding_helper.py      # DeepFace wrapper for facial embeddings
├── index_user_faces.py            # Creates Pinecone index and uploads faces
├── query_user_similarity.py       # Interactive facial similarity search
├── data/
│   └── store/
│       ├── users.json             # 10 users with profile_image_url
│       └── user_images/           # Cached profile images
└── .env                           # Pinecone API key

## Anomaly Detection (Branch: 05-anomaly-detection)

This branch embeds log events as text and uses Pinecone neighbor similarity to flag outliers.

- Files added:
  - log_event_schema.py — `LogEvent` dataclass with `to_text()` and metadata.
  - generate_logs.py — synthetic normal and anomalous events.
  - pinecone_logs_index.py — index utils (create, upsert).
  - detect_anomalies.py — CLI to embed, upsert, and list top anomalies.

### Quick Start

1) Set `PINECONE_API_KEY` in `.env` (or environment). For Pinecone Local, use `pclocal`.

2) Create venv and install deps:

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install sentence-transformers pinecone-python python-dotenv
```

3) Run anomaly detection:

```powershell
python detect_anomalies.py --count 300 --anomaly_ratio 0.15 --top_k 10
```

Notes:
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine).
- Index: `log-anomalies-index` (serverless aws/us-east-1).
- Baseline scoring = 1 − average top-K similarity; refine per service/centroid as desired.

## Understanding Facial Embeddings

### What are Facial Embeddings?

Facial embeddings convert a face into a **128-dimensional vector** that captures unique facial features:

```python
# Face Image → DeepFace → 128-dim vector
[0.234, -0.567, 0.891, 0.123, ..., -0.456]  # 128 numbers
```

**What the numbers represent:**
- Distance between eyes
- Nose shape and position
- Mouth characteristics
- Face shape and proportions
- Jawline structure
- Overall facial geometry

### Why 128 Dimensions?

- **Facenet model standard**: Industry-proven dimensionality
- **Optimal balance**: Enough detail to distinguish faces, not too complex
- **Efficient storage**: Smaller than image models (512-1536 dims)

### Why DeepFace?

- ✅ **Specialized for faces**: 99%+ accuracy on facial recognition benchmarks
- ✅ **Completely offline**: No API calls to external services
- ✅ **Industry standard**: Used in production facial recognition systems
- ✅ **Fast once cached**: Model downloaded once (~350MB), then instant
- ✅ **No authentication needed**: Everything runs locally
- ✅ **Educational value**: Shows specialized embeddings vs. general-purpose (CLIP)

**First Run Example Output:**
```
Loading users...
Loaded 10 users

Initializing DeepFace (Facenet model)...
This may take 2-3 minutes on first run (downloading models ~350MB)...

Generating facial embeddings using DeepFace (offline)...

  Processing 1/10: Alice Johnson...
    ✓ Image cached: data/store/user_images/USER-001.jpg
    ✓ Face embedding generated: 128 dimensions
    
  Processing 2/10: Bob Chen...
    ✓ Image cached: data/store/user_images/USER-002.jpg
    ✓ Face embedding generated: 128 dimensions
    
  [... continues for all 10 users ...]

✓ Generated embeddings for 10 user faces
✓ Upserted batch 1 (10 facial vectors)

============================================================
Face Index Ready!
============================================================
✓ Total facial vectors stored: 10
✓ DeepFace model: Facenet (128 dimensions)
✓ Similarity metric: Cosine
✓ All processing done offline
```

**Subsequent Runs (much faster):**
```
[DeepFace model loads from cache in seconds]
[Images load from data/store/user_images/ directory]
✓ Generated embeddings for 10 users
✓ All processing complete in ~20 seconds
```

### Example Output - Query Similar Users

```powershell
python query_user_similarity.py

=== Facial Similarity Search ===
⚠️  Educational Purpose: Learning vector embeddings and similarity search

1. Find similar users by ID
2. Find similar users by image
3. View all users
4. Exit

Enter your choice: 1

Enter User ID (e.g., USER-001): USER-001

Searching for users similar to USER-001 (Alice Johnson)...

Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. USER-004 (David Kim)
   Segment: tech-professional
   Similarity: 0.8532
   Profile: https://images.unsplash.com/photo-1500648767791...
   
2. USER-006 (Frank Anderson)
   Segment: gadget-lover
   Similarity: 0.8124
   Profile: https://images.unsplash.com/photo-1506794778202...
   
3. USER-008 (Henry Brown)
   Segment: remote-worker
   Similarity: 0.7998
   Profile: https://images.unsplash.com/photo-1507003211169...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Enter your choice: 2

Enter path to image file: my_photo.jpg

Extracting facial embedding from image...
Searching for similar users...

Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. USER-003 (Carol Martinez)
   Segment: eco-conscious
   Similarity: 0.8765
   
2. USER-007 (Grace Lee)
   Segment: wellness-seeker
   Similarity: 0.8234
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
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

---

**Questions?** Refer to the comprehensive `README-FACIAL-SIMILARITY.md` in this branch for additional technical details, troubleshooting guides, and learning resources.

