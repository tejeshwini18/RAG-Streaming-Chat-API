"""
Example client for testing the Streaming Chat API with RAG
"""
import requests
import json
import sys


def add_documents(base_url: str = "http://localhost:8000"):
    """Add sample documents to the vector store"""
    documents = {
        "documents": [
            """
            FastAPI is a modern, fast (high-performance), web framework for building APIs 
            with Python 3.7+ based on standard Python type hints. It's designed to be easy 
            to use and learn, fast to code, ready for production.
            """,
            """
            Retrieval-Augmented Generation (RAG) is a technique that combines the power of 
            retrieval-based and generation-based approaches. It retrieves relevant documents 
            from a knowledge base and uses them as context for generating more accurate and 
            contextual responses.
            """,
            """
            Vector databases store data as high-dimensional vectors, which are mathematical 
            representations of features or attributes. They enable efficient similarity search 
            and are ideal for applications like semantic search, recommendation systems, and RAG.
            """,
            """
            Streaming responses allow applications to display content incrementally as it's 
            generated, rather than waiting for the complete response. This improves user 
            experience by reducing perceived latency.
            """,
            """
            ChromaDB is an open-source vector database designed for AI applications. It's 
            lightweight, easy to use, and can be embedded directly into applications or run 
            as a standalone server.
            """
        ],
        "metadatas": [
            {"source": "fastapi_docs", "topic": "web_framework"},
            {"source": "rag_paper", "topic": "ai_technique"},
            {"source": "vector_db_guide", "topic": "database"},
            {"source": "streaming_guide", "topic": "api_design"},
            {"source": "chromadb_docs", "topic": "vector_db"}
        ]
    }
    
    response = requests.post(
        f"{base_url}/api/v1/documents/add",
        json=documents
    )
    
    if response.status_code == 200:
        print("✅ Documents added successfully!")
        print(f"   Added {response.json()['documents_added']} documents")
    else:
        print(f"❌ Error adding documents: {response.text}")


def chat_streaming(base_url: str = "http://localhost:8000", query: str = None):
    """Chat with streaming response"""
    if query is None:
        query = "What is RAG and how does it work?"
    
    messages = {
        "messages": [
            {"role": "user", "content": query}
        ],
        "stream": True
    }
    
    print(f"\n💬 Query: {query}\n")
    print("🤖 Response (streaming):\n")
    
    response = requests.post(
        f"{base_url}/api/v1/chat",
        json=messages,
        stream=True
    )
    
    if response.status_code != 200:
        print(f"❌ Error: {response.text}")
        return
    
    full_response = ""
    for line in response.iter_lines():
        if line:
            data = line.decode('utf-8')
            if data.startswith('data: '):
                try:
                    content = json.loads(data[6:])
                    if content != "[DONE]":
                        chunk = content.get("content", "")
                        print(chunk, end='', flush=True)
                        full_response += chunk
                except json.JSONDecodeError:
                    pass
    
    print("\n\n" + "="*50)
    print(f"✅ Complete response received ({len(full_response)} characters)")


def chat_non_streaming(base_url: str = "http://localhost:8000", query: str = None):
    """Chat with non-streaming response"""
    if query is None:
        query = "What is FastAPI?"
    
    messages = {
        "messages": [
            {"role": "user", "content": query}
        ],
        "stream": False
    }
    
    print(f"\n💬 Query: {query}\n")
    
    response = requests.post(
        f"{base_url}/api/v1/chat",
        json=messages
    )
    
    if response.status_code == 200:
        result = response.json()
        print("🤖 Response:\n")
        print(result["response"])
        print(f"\n📚 Context used: {result['context_used']}")
    else:
        print(f"❌ Error: {response.text}")


def search_documents(base_url: str = "http://localhost:8000", query: str = "vector database"):
    """Search documents without generating a response"""
    print(f"\n🔍 Searching for: {query}\n")
    
    response = requests.get(
        f"{base_url}/api/v1/documents/search",
        params={"query": query, "top_k": 3}
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"Found {results['count']} relevant documents:\n")
        
        for i, doc in enumerate(results['results'], 1):
            print(f"{i}. Similarity: {doc['similarity']:.3f}")
            print(f"   Content: {doc['content'][:100]}...")
            print(f"   Metadata: {doc.get('metadata', {})}\n")
    else:
        print(f"❌ Error: {response.text}")


def health_check(base_url: str = "http://localhost:8000"):
    """Check API health"""
    response = requests.get(f"{base_url}/api/v1/health")
    
    if response.status_code == 200:
        print("✅ API is healthy!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ API health check failed: {response.text}")


if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print("🚀 Streaming Chat API with RAG - Example Client\n")
    print("="*50)
    
    # Health check
    print("\n1. Health Check")
    health_check(base_url)
    
    # Add documents
    print("\n2. Adding Sample Documents")
    add_documents(base_url)
    
    # Search documents
    print("\n3. Document Search")
    search_documents(base_url, "RAG")
    
    # Chat with streaming
    print("\n4. Streaming Chat")
    chat_streaming(base_url, "Explain RAG in simple terms")
    
    # Chat without streaming
    print("\n5. Non-Streaming Chat")
    chat_non_streaming(base_url, "What is FastAPI?")
