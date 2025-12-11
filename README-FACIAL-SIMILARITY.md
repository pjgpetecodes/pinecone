# Pinecone Vector Search - Facial Similarity

**Branch**: `4-facial-similarity-search`  
**Objective**: Measure and rank facial similarity using Pinecone's vector search to compare embedding distances.

This branch demonstrates **facial similarity measurement** using:
- **Offline facial embeddings** using DeepFace (no API calls, no costs)
- **Pinecone vector similarity search** to rank users by facial distance
- **Cosine similarity** for comparing high-dimensional face vectors
- **Educational context** - Learning vector embedding fundamentals

## Important: Educational Purpose

⚠️ **This is an EDUCATIONAL DEMONSTRATION of vector similarity measurement.**

This implementation:
- ✅ Teaches embedding generation and distance calculation
- ✅ Shows how Pinecone enables similarity search on any embeddings
- ✅ Demonstrates cosine similarity in 128-dimensional space
- ✅ Uses public, diverse profile images (Unsplash)
- ✅ **Completely standalone** - not integrated with recommendations

This is **NOT** intended for:
- Real-world user profiling or identification
- Prediction or discrimination
- Any non-educational use case

## Prerequisites

- Python 3.8 or higher
- DeepFace: `pip install deepface opencv-python-headless`
- Other dependencies: pinecone, pillow, requests, numpy

## Setup

### 1. Create Virtual Environment (if not done)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install pinecone python-dotenv openai deepface opencv-python-headless pillow requests numpy torch torchvision
```

### 3. Configure Environment Variables

Ensure `.env` file exists with:
```env
PINECONE_API_KEY=pclocal
```

### 4. Start Pinecone Local (if not running)

```powershell
docker rm -f pinecone-local
docker run -d --name pinecone-local `
  -e PORT=5081 -e PINECONE_HOST=localhost `
  -p 5081:5081 -p 5082:5082 `
  --platform linux/amd64 `
  ghcr.io/pinecone-io/pinecone-local:latest
```

## Quick Start

### Step 1: Index User Facial Embeddings

```powershell
python index_user_faces.py
```

This will:
1. Load 10 user profiles with profile images
2. Extract facial embeddings using DeepFace (offline)
3. Cache images locally in `data/store/user_images/`
4. Store embeddings in Pinecone `user-faces-index`

**First run**: Downloads DeepFace models (~350MB), takes 2-3 minutes  
**Subsequent runs**: Uses cached models, takes ~30 seconds

**Output:**
```
Loading users...
Loaded 10 users

Generating facial embeddings using DeepFace (offline)...
This may take a minute on first run (downloading models)...

  Processing 1/10: Alice Johnson...
  Processing 2/10: Bob Chen...
  [... more users ...]

✓ Generated embeddings for 10 user faces
✓ Upserted batch 1 (10 facial vectors)

============================================================
Face Index Ready!
============================================================
✓ Total facial vectors stored: 10
✓ Embedding model: Facenet
✓ Embedding dimension: 128
✓ Similarity metric: Cosine
✓ All processing done OFFLINE (no API calls)
```

### Step 2: Search Similar Users

```powershell
python query_user_similarity.py
```

Interactive menu with 4 options:

1. **Find similar users by ID**
   ```
   Choice: 1
   User ID: USER-001
   
   Results:
   1. Carol Martinez (USER-003) - Similarity: 0.8432
   2. Grace Lee (USER-007) - Similarity: 0.8156
   3. Isabella Moore (USER-009) - Similarity: 0.7923
   ```

2. **Find similar users by image**
   ```
   Choice: 2
   Image path: data/store/user_images/USER-001.jpg
   
   Results:
   1. Carol Martinez (USER-003) - Similarity: 0.8432
   2. Grace Lee (USER-007) - Similarity: 0.8156
   ```

3. **View all users**
   - Lists all 10 users with their segments and preferences

4. **Exit**

## Data

### Users (10 profiles with images)
- Located in: `data/store/users.json`
- Each user has: `id`, `name`, `segment`, `profile_image_url`
- Profile images: Cached locally in `data/store/user_images/`

### Facial Embeddings
- **Model**: Facenet (from DeepFace library)
- **Dimension**: 128-dimensional vectors
- **Storage**: Pinecone `user-faces-index`
- **Similarity Metric**: Cosine similarity (0-1 scale)

## How It Works

### Step 1: Image Download & Caching
```
Profile URL → Download → Cache locally → Reuse on future runs
```

### Step 2: Facial Embedding Extraction
```
Image → DeepFace (Facenet) → 128-dim vector
Example: [0.234, -0.156, 0.789, ..., 0.123]
```

### Step 3: Pinecone Indexing
```
User ID: USER-001-face
Embedding: [0.234, -0.156, 0.789, ...]
Metadata: {name, segment, profile_image_url, ...}
```

### Step 4: Similarity Search
```
Query User Embedding
         ↓
Pinecone finds closest vectors by cosine distance
         ↓
Returns ranked list of similar users
```

### Vector Similarity Calculation

Cosine similarity measures angle between vectors (not magnitude):
```
similarity = (v1 · v2) / (||v1|| × ||v2||)

Range: 0 to 1
  1.0 = identical vectors (same face)
  0.8+ = very similar facial features
  0.5-0.8 = moderately similar
  <0.5 = dissimilar
```

## File Structure

```
.
├── data/store/
│   ├── users.json                    # User profiles with profile_image_url
│   └── user_images/                  # Cached profile photos
│       ├── USER-001.jpg
│       ├── USER-002.jpg
│       └── ... (10 total)
├── face_embedding_helper.py          # DeepFace wrapper (offline)
├── index_user_faces.py               # Create facial embeddings index
├── query_user_similarity.py          # Interactive similarity search
└── README.md                         # This file
```

## Key Concepts Demonstrated

### 1. Face Embeddings
Facial embeddings are learned representations capturing key facial characteristics:
- Face shape, symmetry
- Eye, nose, mouth positions
- Skin tone and texture
- Facial landmarks

Modern face recognition models (like Facenet) convert faces to 128-dimensional vectors where similar faces cluster together.

### 2. Vector Distance Metrics
**Cosine Similarity** (used here):
- Measures angle between vectors, not magnitude
- Ideal for high-dimensional embeddings
- Scale: 0 (perpendicular) to 1 (identical)
- Robust to vector magnitude differences

**Euclidean Distance** (alternative):
- Measures straight-line distance
- More sensitive to magnitude

### 3. Pinecone Similarity Search
Pinecone efficiently finds nearest neighbors in high-dimensional space:

```python
# Query: Find top-5 users most similar to USER-001
results = index.query(
    vector=user_embedding,      # 128-dim vector
    top_k=5,                    # Return 5 results
    include_metadata=True        # Include user info
)
```

### 4. Offline Processing
All facial recognition happens locally:
- ✅ DeepFace models cached on disk
- ✅ Image processing on your machine
- ✅ No API calls (except Pinecone for search)
- ✅ No data sent to external services

## Understanding Facial Embeddings

### Why 128 Dimensions?

Facenet produces 128-dimensional embeddings because:
- Enough dimensions to capture facial nuances
- Compact enough for efficient computation
- Trained on millions of face pairs to maximize discriminability

### Why Cosine Similarity?

For face embeddings, cosine similarity works better than Euclidean distance:
- **Normalized**: Embeddings already normalized to unit length
- **Angle-based**: Measures feature directions, not magnitudes
- **Robust**: Works well for high-dimensional sparse data

### How Similar Are Similar?

In facial embedding space:
- **0.90+**: Likely the same person with different expressions
- **0.80-0.89**: Very similar facial features, could be relatives
- **0.70-0.79**: Noticeably similar facial structure
- **0.60-0.69**: Some similar features but clearly different people
- **<0.60**: Distinctly different facial characteristics

## Troubleshooting

### "DeepFace not installed"
```powershell
pip install deepface opencv-python-headless
```

### "No face detected in image"
- Image quality too low
- Face partially obscured
- Face too small in image
- Set `enforce_detection=False` in code (already done)

### "Index not found"
```powershell
python index_user_faces.py
```

### "Connection errors"
Ensure Pinecone container is running:
```powershell
docker ps | grep pinecone-local
```

### Slow first run
DeepFace downloads ~350MB of model files on first use. Subsequent runs use cache.

## How This Relates to Vector Search

This branch demonstrates **vector similarity fundamentals**:

| Concept | Implementation |
|---------|-----------------|
| Embedding generation | DeepFace Facenet (128-dim) |
| Vector storage | Pinecone `user-faces-index` |
| Similarity metric | Cosine distance |
| Search type | Nearest neighbors (top-k) |
| Query | User ID or image path |
| Results | Ranked by embedding distance |

Same concepts apply to:
- **Semantic search** (text embeddings)
- **Product recommendations** (product embeddings)
- **Image search** (visual embeddings)
- **Any embeddings** (audio, video, multimodal)

## Related Branches

| Branch | Objective |
|--------|-----------|
| `main` | Basic RAG - PDF document search |
| `2-recommender-systems` | Personalized product recommendations with metadata filtering |
| `3-multimodal-search` | Hybrid text + image search using CLIP |
| **`4-facial-similarity-search`** | **Facial similarity ranking using embeddings** |

## Learning Objectives

After this branch, you understand:
1. ✅ How to generate embeddings from images (face detection)
2. ✅ How to store embeddings in Pinecone
3. ✅ How to query for similar embeddings (nearest neighbors)
4. ✅ How cosine similarity measures embedding distance
5. ✅ How vector search scales to billions of vectors

## Next Steps

**Try these explorations:**

1. **Compare embedding spaces**
   - Query the same user multiple times
   - Note consistency of results
   
2. **Understand similarity scores**
   - Look at 0.90+ scores (very similar)
   - Look at 0.60-0.70 scores (moderately similar)
   - Relate to actual face photos

3. **Experiment with images**
   - Try different image formats
   - Test with images of same person
   - Test with images of different people

4. **Technical deep-dives**
   - Examine raw embedding vectors (first 5 values printed)
   - Calculate similarities manually using numpy
   - Test different DeepFace models (VGG-Face, Facenet512, etc.)

---

**Questions?** This implementation prioritizes learning over production use. All code is annotated with educational context.
