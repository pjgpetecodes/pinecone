# Pinecone Vector Search with Azure OpenAI Embeddings

A Python application that demonstrates vector search using Pinecone Local and Azure OpenAI embeddings. This project extracts text from PDF files, generates embeddings, and enables semantic search capabilities.

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
pip install pinecone python-dotenv openai pypdf
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

## Running the Application

### Step 1: Extract PDF and Create Index

Run the indexing script:

```powershell
python pinecone_indexes.py
```

This will:
1. Extract paragraphs from the PDF
2. Generate embeddings using Azure OpenAI
3. Create a Pinecone index
4. Upsert all vectors with metadata

**Output:**
```
Extracted 150 paragraphs from XYZ-2021-Annual_Report.pdf
Company: XYZ, Year: 2021

Generating embeddings...
Generated 150 embeddings

Upserted batch 1 (100 vectors)
Upserted batch 2 (50 vectors)

Total vectors upserted: 150

Indexes:
  - example-index
```

### Step 2: Query the Index

Run the query interface:

```powershell
python query_index.py
```

This opens an interactive prompt where you can search:

```
Pinecone Query Interface
==================================================
Index: example-index
Top K Results: 5
==================================================

Enter your query (or 'quit' to exit): What are the company's financial results?
```

**Results display:**
- All 5 matching results with similarity scores (0-1)
- Full metadata: title, company, year, location, file name
- Complete content text
- **Best Match Summary** showing the highest-scoring result

Example output:
```
Found 5 results:

--- Result 1 (Score: 0.8234) ---
Title: Page 3 - Paragraph 1
Company: XYZ
Year: 2021
Location: Finance
File: XYZ-2021-Annual_Report.pdf
Content:
[Full paragraph text...]

=== Best Match Summary ===
Title: Page 3 - Paragraph 1
Score: 0.8234
Company: XYZ
...
```

Type `quit` to exit.

## Project Structure

```
pinecone1/
├── .venv/                          # Virtual environment
├── .env                            # Environment variables (add to .gitignore)
├── .gitignore
├── README.md
├── pinecone_indexes.py             # Main indexing script
├── query_index.py                  # Query interface
├── pdf_helper.py                   # PDF extraction utilities
├── embedding_helper.py             # Azure OpenAI embedding utilities
├── XYZ-2021-Annual_Report.pdf      # Sample PDF file
└── __pycache__/
```

## File Descriptions

### `pinecone_indexes.py`
- Extracts paragraphs from PDF files
- Generates embeddings for each paragraph
- Creates/recreates the Pinecone index
- Upserts vectors with metadata (company, year, location, etc.)

### `query_index.py`
- Interactive query interface
- Converts user queries to embeddings
- Searches Pinecone index
- Displays results with similarity scores

### `pdf_helper.py`
- Extracts text from PDF files
- Chunks text into paragraphs
- Handles filename parsing for metadata
- Manages large content (>7000 characters)

### `embedding_helper.py`
- Initializes Azure OpenAI client
- Generates single embeddings
- Batches embeddings for efficiency

## Troubleshooting

### Docker Container Won't Start

**Error:** `Ports are not available`

**Solution:** Stop conflicting containers:
```powershell
docker ps  # Find the container using ports 5081-5082
docker stop <container-id>
docker rm -f pinecone-local
```

Then restart with the commands above.

### Connection Refused on Port 5082

**Error:** `failed to connect to all addresses; last error: UNAVAILABLE: ipv4:127.0.0.1:5082`

**Solution:** Ensure both ports are exposed:
```powershell
docker rm -f pinecone-local
docker run -d --name pinecone-local `
  -e PORT=5081 -e PINECONE_HOST=localhost `
  -p 5081:5081 -p 5082:5082 `
  --platform linux/amd64 `
  ghcr.io/pinecone-io/pinecone-local:latest
```

### Azure OpenAI Authentication Error

**Error:** `AuthenticationError` or `Invalid API key`

**Solution:**
1. Verify `.env` file has correct values
2. Check Azure Portal for correct instance name and API version
3. Ensure deployment names match your Azure configuration

### No Results Found

**Possible causes:**
- Index is empty (run `pinecone_indexes.py` first)
- PDF file not found
- Query is too different from document content

**Solution:** Test with a specific phrase from your PDF.

## Configuration Options

### Modify Top K Results

In `query_index.py`:
```python
top_k = 10  # Change from 5 to 10
```

### Adjust Batch Size

In `pinecone_indexes.py`:
```python
batch_size = 50  # Change from 100 for slower connections
```

### Change Index Dimensions

Update in `pinecone_indexes.py`:
```python
model_dimensions = 1536  # Must match your embedding model
```

## Performance Tips

1. **Batch Processing:** The application processes embeddings in batches for efficiency
2. **Delay Between Batches:** 1-second delay prevents overwhelming local Pinecone
3. **Top K Results:** Limit results to what you need (higher K = slower queries)
4. **PDF Size:** Test with smaller PDFs first before processing large documents

## Next Steps

- Integrate with a web application (Flask/FastAPI)
- Add filtering by metadata (company, year, location)
- Implement reranking for better results
- Add authentication and user sessions
- Deploy Pinecone to production (Azure/AWS)

## Additional Resources

- [Pinecone Documentation](https://docs.pinecone.io)
- [Pinecone Local Setup](https://docs.pinecone.io/guides/projects/pinecone-local)
- [Azure OpenAI Embeddings](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#embeddings)
- [Vector Search Concepts](https://docs.pinecone.io/learn/vector-search)

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review Docker logs: `docker logs pinecone-local`
3. Verify `.env` configuration
4. Check Azure OpenAI credentials
