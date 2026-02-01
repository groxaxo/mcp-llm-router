"""
Example: Using Local Cross-Encoder Reranker for Memory Search

This example demonstrates how to use the local Qwen-based cross-encoder reranker
to improve semantic search results in the MCP LLM Router.
"""

import asyncio
from mcp_llm_router.memory import (
    MemoryStore,
    MemorySettings,
    EmbeddingConfig,
    RerankConfig,
    embed_texts,
    rerank_documents,
)


async def example_local_reranking():
    """Example showing local reranker configuration and usage."""
    
    print("=" * 70)
    print("Local Cross-Encoder Reranking Example")
    print("=" * 70)
    print()
    
    # 1. Configure memory settings with local reranking
    print("1. Configure memory settings with local reranking:")
    print()
    
    memory_settings = MemorySettings(
        embedding=EmbeddingConfig(
            provider="ollama",  # Use local Ollama for embeddings
            base_url="http://localhost:11434",
            model="qwen3-embedding:0.6b",
        ),
        rerank=RerankConfig(
            provider="local",  # Use local cross-encoder
            mode="local",      # Use local mode (not API or LLM)
            model="tomaarsen/Qwen3-Reranker-0.6B-seq-cls",  # HuggingFace model
        ),
    )
    
    print(f"  Embedding Provider: {memory_settings.embedding.provider}")
    print(f"  Embedding Model: {memory_settings.embedding.model}")
    print(f"  Rerank Provider: {memory_settings.rerank.provider}")
    print(f"  Rerank Mode: {memory_settings.rerank.mode}")
    print(f"  Rerank Model: {memory_settings.rerank.model}")
    print()
    
    # 2. Example documents (simulating memory search results)
    print("2. Example documents:")
    print()
    
    query = "What are the key features of MCP?"
    
    documents = [
        {
            "doc_id": "doc1",
            "content": "The Model Context Protocol enables seamless communication between AI assistants and external tools, providing a standardized interface for tool integration.",
            "score": 0.75,
        },
        {
            "doc_id": "doc2",
            "content": "Python is a versatile programming language widely used in web development, data science, and automation tasks.",
            "score": 0.65,
        },
        {
            "doc_id": "doc3",
            "content": "MCP features include multi-provider routing, session management, local-first memory with embeddings, and quality gating with judge tools.",
            "score": 0.80,
        },
        {
            "doc_id": "doc4",
            "content": "Machine learning models can be trained on large datasets to recognize patterns and make predictions.",
            "score": 0.60,
        },
        {
            "doc_id": "doc5",
            "content": "The MCP Router supports cross-server tool calling and works with any MCP-compatible client, not just specific IDEs.",
            "score": 0.70,
        },
    ]
    
    print(f"Query: {query}")
    print(f"Number of documents: {len(documents)}")
    print()
    
    # 3. Show original ranking (by embedding similarity score)
    print("3. Original ranking (by embedding similarity score):")
    print()
    
    for i, doc in enumerate(sorted(documents, key=lambda d: d['score'], reverse=True), 1):
        print(f"{i}. [Score: {doc['score']:.2f}] {doc['content'][:80]}...")
    print()
    
    # 4. Apply local reranking
    print("4. After local cross-encoder reranking:")
    print()
    
    try:
        reranked = await rerank_documents(query, documents, memory_settings.rerank)
        
        if reranked:
            for i, doc in enumerate(reranked, 1):
                rerank_score = doc.get('rerank_score', 'N/A')
                if isinstance(rerank_score, float):
                    print(f"{i}. [Rerank: {rerank_score:.4f}] {doc['content'][:80]}...")
                else:
                    print(f"{i}. [Rerank: {rerank_score}] {doc['content'][:80]}...")
        else:
            print("  (Reranking not available - requires PyTorch, transformers, and network access)")
            print("  Returned original ranking")
        
    except Exception as e:
        print(f"  Error during reranking: {e}")
        print("  This is expected if PyTorch/transformers are not installed")
    
    print()
    print("=" * 70)
    print("Configuration Summary")
    print("=" * 70)
    print()
    print("To use local cross-encoder reranking in production, set:")
    print()
    print("  export RERANK_PROVIDER='local'")
    print("  export RERANK_MODE='local'")
    print("  export RERANK_MODEL='tomaarsen/Qwen3-Reranker-0.6B-seq-cls'")
    print()
    print("Requirements:")
    print("  - pip install torch")
    print("  - pip install transformers")
    print("  - Internet access for first model download (~1.2GB)")
    print()
    print("Benefits of local reranking:")
    print("  ✓ No external API calls - complete privacy")
    print("  ✓ No API costs")
    print("  ✓ Faster after initial model load")
    print("  ✓ More accurate relevance ranking than simple cosine similarity")
    print("  ✓ Works offline after initial download")
    print()


if __name__ == "__main__":
    asyncio.run(example_local_reranking())
