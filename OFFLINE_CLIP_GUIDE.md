# Offline CLIP for Multimodal Search

## Quick Reference

### What is CLIP?
**CLIP** (Contrastive Language-Image Pre-training) is an open-source model from OpenAI that understands both text and images. It can:
- Convert images to embeddings (512 dimensions)
- Convert text to embeddings in the same semantic space
- Compare images and text using cosine similarity
- Run completely offline on your machine

### Why Offline CLIP Instead of API?
| Aspect | Offline CLIP | Azure OpenAI Vision |
|--------|--------------|---------------------|
| API Key Required | ❌ No | ✅ Yes |
| Cost | ✅ Free | ❌ Pay per call |
| Network Required | ❌ (after cached) | ✅ Always |
| Rate Limiting | ❌ No | ✅ Yes |
| Privacy | ✅ On your machine | ❌ To Azure |
| Speed | ✅ Fast | ⚠️ Slower |
| Offline Support | ✅ Yes (cached) | ❌ No |

### Installation

```powershell
pip install torch transformers pillow requests
```

### First Run

The first time you run `index_images.py`, it will download the CLIP model (~350MB):

```powershell
python index_images.py
```

**First run output:**
```
Loading CLIP model: openai/clip-vit-base-patch32
Using device: cuda
CLIP model loaded. Embedding dimension: 512

Initializing CLIP model for offline image embeddings...
✓ CLIP initialized. Embedding dimension: 512

Loading products...
Loaded 20 products

Generating image embeddings using CLIP (offline)...
Processing images - this may take a few minutes on first run (downloading model cache)...

  Processing PROD-001: Wireless Bluetooth Headphones...
  ✓ Generated embeddings for 20 product images
  ✓ Upserted batch 1 (20 image vectors)

Image indexing complete!
```

### Subsequent Runs

After the model is cached, subsequent runs are much faster (30 seconds instead of 2-5 minutes):

```powershell
python index_images.py
```

The cached model is stored in: `~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/`

### GPU Support

If you have NVIDIA CUDA installed, CLIP will use GPU for faster processing:

```powershell
# Install GPU version of PyTorch (optional, faster)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Check device being used:
```python
import torch
print(torch.cuda.is_available())  # True if GPU available
print(torch.cuda.get_device_name(0))  # GPU name
```

### Model Variants

You can use different CLIP models by modifying `image_helper.py`:

```python
# Smaller, faster model (512-dim embeddings) - DEFAULT
ImageHelper(model_name="openai/clip-vit-base-patch32")

# Larger, more accurate model (768-dim embeddings)
ImageHelper(model_name="openai/clip-vit-large-patch14")
```

### How Image Search Works

1. **Download image** from URL
2. **Load CLIP model** from cache
3. **Process image** through CLIP encoder
4. **Generate embedding** (512-dimensional vector)
5. **Store in Pinecone** with ID `PROD-XXX-image`
6. **Query using** same CLIP model

### Hybrid Text + Image Search

Text embeddings (1536-dim from Azure OpenAI) and image embeddings (512-dim from CLIP) work together:

```python
# Text query
text_query = "wireless headphones"
text_embedding = text_helper.get_embedding(text_query)  # 1536-dim

# Image query  
image_url = "https://unsplash.com/..."
image_embedding = image_helper.get_image_embedding(image_url)  # 512-dim

# Both search Pinecone independently, then results are combined
combined_results = hybrid_search(text_embedding, image_embedding)
```

### Troubleshooting

#### Model Download Fails
```powershell
# Clear cache and retry
Remove-Item $env:USERPROFILE\.cache\huggingface -Recurse -Force
python index_images.py
```

#### Out of Memory (OOM) Error
**Solution 1:** Use smaller model variant
```python
ImageHelper(model_name="openai/clip-vit-base-patch32")  # Smaller
```

**Solution 2:** Process fewer images at once
```python
# Modify index_images.py to batch process
batch_size = 5  # Instead of all 20 at once
```

**Solution 3:** Use CPU instead of GPU
```python
self.device = "cpu"  # Force CPU in image_helper.py
```

#### Internet Required for First Run
Yes, CLIP model download requires internet. But after download, everything works offline!

#### Different Results Each Time
Expected! CLIP embeddings are deterministic, but search results depend on Pinecone's query behavior and other factors.

### Performance Metrics

| Operation | Time | Notes |
|-----------|------|-------|
| CLIP model download | 2-3 min | One-time, ~350MB |
| Model load from cache | <1 sec | Very fast |
| Image download (20 products) | 5-10 sec | Depends on internet |
| Image processing (20 products) | 30 sec | First run; 10 sec cached |
| Pinecone upsert | 5 sec | 20 vectors in 1 batch |
| **Total first run** | **2-5 min** | Mostly model download |
| **Total subsequent runs** | **30-60 sec** | Just processing |

### Comparison with JavaScript (transformers.js)

This Python implementation is equivalent to Pinecone's JavaScript example using `transformers.js` CLIP:
- Same CLIP model
- Same offline capability
- Same output dimensions (512 or 768)
- Python for server-side, JS for browser-side

### References

- **CLIP Paper:** https://arxiv.org/abs/2103.14030
- **HuggingFace CLIP:** https://huggingface.co/openai/clip-vit-base-patch32
- **Transformers Library:** https://huggingface.co/docs/transformers/
- **PyTorch:** https://pytorch.org/

---

**Key Takeaway:** CLIP gives you powerful image understanding completely for free, with no API calls, no costs, and no rate limits. Perfect for local development and privacy-conscious applications!
