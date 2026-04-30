# RAG-Streaming-Chat-API

A **streaming chat API** with **Retrieval-Augmented Generation (RAG)** — production-ready, modular, and easy to extend.

## Overview

This project provides:

- **Streaming responses** via Server-Sent Events (SSE)
- **RAG pipeline**: embed query → retrieve docs → inject context → stream LLM output
- **Flexible backends**: OpenAI or local [Ollama](https://ollama.com) for the LLM; OpenAI or local Sentence Transformers for embeddings
- **Optional reranker** (cross-encoder) for better retrieval quality
- **FastAPI** app with CORS, health checks, and a simple chat UI

## Quick Start

```bash
cd streaming_chat_api_with_RAG
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set OPENAI_API_KEY or USE_OLLAMA=true
python main.py
```

- **API**: [http://localhost:8000](http://localhost:8000)  
- **Chat UI**: [http://localhost:8000/](http://localhost:8000/)  
- **Health**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

## Project Layout

```
RAG-Streaming-Chat-API/
├── README.md                          # This file
└── streaming_chat_api_with_RAG/       # Main application
    ├── api/routes.py                  # Chat, documents, health, models
    ├── rag/                           # Embedder, vector store, retriever, reranker
    ├── llm/streaming_llm.py           # OpenAI / Ollama streaming
    ├── config.py
    ├── main.py
    ├── requirements.txt
    ├── .env.example
    └── README.md                      # Full docs, API reference, examples
```

## Documentation

For **setup, API reference, RAG flow, configuration, and usage examples**, see:

**[streaming_chat_api_with_RAG/README.md](streaming_chat_api_with_RAG/README.md)**


## GitHub Actions Deployment

This repo now includes:

- `Dockerfile` for containerized runtime
- `.github/workflows/deploy.yml` to build/push to GHCR on every push to `main`
- Optional automatic deploy trigger to Render via webhook

### Required GitHub Secrets

Add these in **Settings → Secrets and variables → Actions**:

- `RENDER_DEPLOY_HOOK_URL` (optional): Render deploy webhook URL. If omitted, workflow only builds/pushes image.

### Image Output

The workflow pushes:

- `ghcr.io/<owner>/<repo>:latest`
- `ghcr.io/<owner>/<repo>:sha-<commit>`

Use one of these tags in your hosting platform.

## License

MIT License
