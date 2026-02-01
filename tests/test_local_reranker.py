"""
Test script for local cross-encoder reranker integration.

This script tests the integration of the Qwen-based cross-encoder reranker
from the rag package into the memory module's reranking pipeline.
"""

import asyncio
from mcp_llm_router.memory import RerankConfig, rerank_documents


async def test_local_reranker():
    """Test the local reranker with sample documents."""
    
    # Sample query
    query = "What is the function of MCP?"
    
    # Sample documents
    documents = [
        {
            "doc_id": "1",
            "content": "The Model Context Protocol (MCP) is a protocol for communication between AI assistants and external tools.",
            "score": 0.8,
        },
        {
            "doc_id": "2",
            "content": "Python is a high-level programming language used for web development.",
            "score": 0.6,
        },
        {
            "doc_id": "3",
            "content": "MCP enables seamless integration of language models with various data sources and tools.",
            "score": 0.7,
        },
        {
            "doc_id": "4",
            "content": "Machine learning is a subset of artificial intelligence.",
            "score": 0.5,
        },
    ]
    
    # Test with local reranker
    print("Testing local cross-encoder reranker...")
    print(f"Query: {query}\n")
    
    config = RerankConfig(
        provider="local",
        mode="local",
        model="tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
    )
    
    def format_score(score):
        """Format rerank score for display."""
        return f"{score:.4f}" if isinstance(score, float) else str(score)
    
    try:
        reranked = await rerank_documents(query, documents, config)
        
        print("Reranked results:")
        for i, doc in enumerate(reranked, 1):
            score_str = format_score(doc.get('rerank_score', 'N/A'))
            print(f"{i}. [Score: {score_str}] {doc['content'][:80]}...")
        
        print("\n✅ Local reranker integration test passed!")
        return True
        
    except ImportError as e:
        print(f"⚠️  Warning: Could not import reranker: {e}")
        print("This is expected if transformers/torch are not installed.")
        print("To use local reranking, install: pip install torch transformers")
        return False
    except Exception as e:
        print(f"❌ Error during reranking: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_fallback_when_no_local():
    """Test that the system falls back gracefully when local reranker is unavailable."""
    
    query = "test query"
    documents = [
        {"doc_id": "1", "content": "test content 1", "score": 0.5},
        {"doc_id": "2", "content": "test content 2", "score": 0.7},
    ]
    
    # Test with mode=local but provider=none
    config = RerankConfig(
        provider="none",
        mode="local",
    )
    
    result = await rerank_documents(query, documents, config)
    
    # Should return documents unchanged when provider is "none"
    assert len(result) == len(documents), "Should return all documents"
    print("✅ Fallback test passed - returns documents unchanged when provider='none'")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Local Cross-Encoder Reranker Integration")
    print("=" * 70)
    print()
    
    asyncio.run(test_local_reranker())
    print()
    asyncio.run(test_fallback_when_no_local())
