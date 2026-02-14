# Distributed LLM + Streaming Chat API with RAG

A production-ready streaming chat API with Retrieval-Augmented Generation (RAG) capabilities.

## 🧠 Architecture

Instead of the simple flow:
```
User → LLM → Response
```

This system implements:
```
User → Embed Query → Retrieve Docs → Inject Context → LLM → Stream Output
```

## 🚀 Features

- **Streaming Responses**: Real-time token streaming using Server-Sent Events (SSE)
- **RAG Pipeline**: 
  - Query embedding (OpenAI or local Sentence Transformers)
  - Vector-based document retrieval (NumPy-based store; no C++ build required)
  - Context injection into LLM prompts
- **Distributed Ready**: Modular architecture for scaling components
- **FastAPI**: Modern async API framework
- **Flexible Embeddings**: Support for OpenAI embeddings or local models

## 📁 Project Structure

```
streaming_chat_api_with_RAG/
├── api/
│   ├── __init__.py
│   └── routes.py          # API endpoints
├── rag/
│   ├── __init__.py
│   ├── embedder.py        # Text embedding service
│   ├── vector_store.py    # NumPy-based vector store (persisted to disk)
│   └── retriever.py       # Document retrieval logic
├── llm/
│   ├── __init__.py
│   └── streaming_llm.py   # LLM streaming integration
├── config.py              # Configuration management
├── main.py                # FastAPI application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── README.md
```

## 🛠️ Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your settings:

```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 3. Run the Server

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📡 API Endpoints

### POST `/api/v1/chat`

Streaming chat endpoint with RAG.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "What is machine learning?"}
  ],
  "stream": true
}
```

**Response (Streaming):**
```
data: {"content": "Machine"}
data: {"content": " learning"}
data: {"content": " is..."}
data: [DONE]
```

### POST `/api/v1/documents/add`

Add documents to the vector store.

**Request:**
```json
{
  "documents": [
    "Machine learning is a subset of artificial intelligence...",
    "Deep learning uses neural networks..."
  ],
  "metadatas": [
    {"source": "ml_intro.txt"},
    {"source": "dl_basics.txt"}
  ]
}
```

### GET `/api/v1/documents/search?query=neural networks&top_k=5`

Search documents without generating a response.

### GET `/api/v1/health`

Health check endpoint.

## 🔄 RAG Pipeline Flow

1. **User Query**: User sends a message
2. **Embed Query**: Convert query text to vector embedding
3. **Retrieve Docs**: Find similar documents using vector similarity search
4. **Format Context**: Combine retrieved documents into context string
5. **Inject Context**: Add context to LLM system prompt
6. **Stream Response**: LLM generates response token by token

## 💡 Usage Example

### Python Client

```python
import requests
import json

# Add documents
documents = {
    "documents": [
        "FastAPI is a modern web framework for building APIs...",
        "RAG combines retrieval and generation for better answers..."
    ]
}
requests.post("http://localhost:8000/api/v1/documents/add", json=documents)

# Chat with streaming
messages = {
    "messages": [
        {"role": "user", "content": "Tell me about FastAPI"}
    ],
    "stream": True
}

response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json=messages,
    stream=True
)

for line in response.iter_lines():
    if line:
        data = line.decode('utf-8')
        if data.startswith('data: '):
            content = json.loads(data[6:])
            if content != "[DONE]":
                print(content["content"], end='', flush=True)
```

### JavaScript/TypeScript Client

```javascript
const response = await fetch('http://localhost:8000/api/v1/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    messages: [{ role: 'user', content: 'What is RAG?' }],
    stream: true
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      if (data !== '[DONE]') {
        process.stdout.write(data.content);
      }
    }
  }
}
```

## ⚙️ Configuration

Key configuration options in `.env`:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: Model to use (default: gpt-4o; or use gpt-4o-mini, gpt-3.5-turbo)
- `EMBEDDING_MODEL`: Local embedding model (default: all-MiniLM-L6-v2)
- `USE_OPENAI_EMBEDDINGS`: Set to `true` to use OpenAI embeddings
- `TOP_K_RESULTS`: Number of documents to retrieve (default: 5)
- `SIMILARITY_THRESHOLD`: Minimum similarity score (default: 0.7)

## 🏗️ Architecture Benefits

- **Modular**: Each component (embedder, retriever, LLM) is independent
- **Scalable**: Can distribute components across services
- **Flexible**: Easy to swap embedding models or vector stores
- **Production-Ready**: Includes error handling, health checks, and proper async patterns

## 🔧 Extending the System

### Add Custom Embedder

```python
class CustomEmbedder(Embedder):
    def embed_query(self, text: str) -> List[float]:
        # Your custom embedding logic
        pass
```

### Add Custom Vector Store

```python
class CustomVectorStore(VectorStore):
    def query(self, query_text: str, n_results: int = 5):
        # Your custom retrieval logic
        pass
```

### Add Custom LLM Provider

```python
class CustomLLM(StreamingLLM):
    async def stream_response(self, messages, context):
        # Your custom streaming logic
        pass
```

## 📝 License

MIT License
