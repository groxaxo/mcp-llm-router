"""
Example script demonstrating how to index a repository and perform a retrieval.

This script scans the current working directory, indexes source files
and markdown files into a Chroma vector store using the configured embedding,
and then allows the user to enter queries to retrieve relevant code/doc
fragments.  It is intended as a standalone demonstration and can be adapted
for integration into larger systems.
"""
import argparse

from .indexer import index_path
from .retriever import retrieve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default=".",
        help="Root directory to index (defaults to current working dir)",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default=".py,.md,.txt",
        help="Comma-separated list of file extensions to index",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive query mode after indexing",
    )
    args = parser.parse_args()

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    stats = index_path(args.path, exts=exts)
    print(f"Indexed {stats['files_indexed']} files into {stats['chunks_indexed']} chunks.")

    if args.interactive:
        try:
            while True:
                q = input("Query (or 'exit'): ").strip()
                if not q or q.lower() == "exit":
                    break
                hits = retrieve(q, top_k=5)
                for i, hit in enumerate(hits):
                    print(f"\nResult {i+1} (distance {hit['distance']:.4f}):")
                    print(f"ID: {hit['id']}")
                    print(f"Path: {hit['meta'].get('path')}")
                    print(f"Chunk index: {hit['meta'].get('chunk_index')}")
                    print(f"Content:\n{hit['doc']}\n")
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()