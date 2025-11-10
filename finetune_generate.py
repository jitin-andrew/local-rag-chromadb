import argparse
import os
from typing import List, Optional, Tuple
import re

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


def read_lines(path: str) -> List[str]:
    """Read lines from a file, filtering out empty lines."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return [line for line in lines if line]


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for better retrieval."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def build_knowledge_base(
    documents: List[str],
    vector_db_path: str = "./chroma_db",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 200,
    chunk_overlap: int = 50,
):
    """Build a vector database from documents."""
    print(f"Loading embedding model: {embedding_model}")
    embedding_model_obj = SentenceTransformer(embedding_model)
    
    # initialize ChromaDB
    client = chromadb.PersistentClient(
        path=vector_db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # delete collection if it exists, then create a new one
    try:
        client.delete_collection(name="knowledge_base")
    except Exception:
        pass  # collection doesn't exist, which is fine
    
    # create new collection
    collection = client.create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )
    
    # process documents
    all_chunks = []
    chunk_metadata = []
    
    for doc_idx, doc in enumerate(documents):
        chunks = chunk_text(doc, chunk_size=chunk_size, overlap=chunk_overlap)
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({
                "doc_id": str(doc_idx),
                "chunk_id": str(chunk_idx),
            })
    
    if not all_chunks:
        raise ValueError("No text chunks created from documents.")
    
    print(f"Creating embeddings for {len(all_chunks)} chunks...")
    embeddings = embedding_model_obj.encode(
        all_chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    
    # add to ChromaDB
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    collection.add(
        embeddings=embeddings.tolist(),
        documents=all_chunks,
        ids=ids,
        metadatas=chunk_metadata,
    )
    
    print(f"Knowledge base created with {len(all_chunks)} chunks in {vector_db_path}")
    return embedding_model_obj, collection


def retrieve_relevant_context(
    query: str,
    embedding_model: SentenceTransformer,
    collection,
    top_k: int = 3,
) -> str:
    """Retrieve relevant context from the knowledge base."""
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )
    
    if results["documents"] and len(results["documents"][0]) > 0:
        contexts = results["documents"][0]
        return "\n\n".join(contexts)
    return ""


def generate_with_rag(
    query: str,
    vector_db_path: str,
    # model_name: str = "microsoft/phi-2",
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    retrieval_top_k: int = 3,
    use_rag: bool = True,
):
    """Generate text using RAG (Retrieval-Augmented Generation)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load embedding model and vector DB
    if use_rag:
        print(f"Loading embedding model: {embedding_model_name}")
        embedding_model = SentenceTransformer(embedding_model_name)
        
        client = chromadb.PersistentClient(
            path=vector_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(name="knowledge_base")
        
        # retrieve relevant context
        print(f"Retrieving relevant context for: {query}")
        context = retrieve_relevant_context(
            query, embedding_model, collection, top_k=retrieval_top_k
        )
        
        if context:
            # create RAG prompt
            prompt = f"""Context information:
{context}

Question: {query}

Answer based on the context above:"""
        else:
            print("Warning: No relevant context found. Generating without RAG.")
            prompt = f"Question: {query}\nAnswer:"
    else:
        prompt = f"Question: {query}\nAnswer:"
    
    # load generation model
    print(f"Loading generation model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        device_map="auto" if device.type == "cuda" else None,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # extract only the answer part (after the prompt)
    if prompt in generated_text:
        answer = generated_text.split(prompt)[-1].strip()
    else:
        answer = generated_text
    
    return answer, prompt if use_rag else None


def ingest_documents(
    input_file: str,
    vector_db_path: str = "./chroma_db",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 200,
    chunk_overlap: int = 50,
):
    """Ingest documents into the vector database."""
    print(f"Reading documents from: {input_file}")
    documents = read_lines(input_file)
    
    if not documents:
        raise ValueError("No non-empty lines found in input file.")
    
    print(f"Processing {len(documents)} documents...")
    build_knowledge_base(
        documents=documents,
        vector_db_path=vector_db_path,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    print("Document ingestion complete!")


def main():
    parser = argparse.ArgumentParser(
        description="RAG system for minimal text documents with local vector DB",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # ingest subcommand
    p_ingest = subparsers.add_parser(
        "ingest", help="Ingest documents into the vector database"
    )
    p_ingest.add_argument(
        "--input_file", type=str, required=True, help="Path to text file with documents"
    )
    p_ingest.add_argument(
        "--vector_db_path", type=str, default="./chroma_db", help="Path to vector DB"
    )
    p_ingest.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name",
    )
    p_ingest.add_argument(
        "--chunk_size", type=int, default=200, help="Chunk size for text splitting"
    )
    p_ingest.add_argument(
        "--chunk_overlap", type=int, default=50, help="Overlap between chunks"
    )
    
    # query subcommand
    p_query = subparsers.add_parser("query", help="Query the RAG system")
    p_query.add_argument(
        "--vector_db_path", type=str, default="./chroma_db", help="Path to vector DB"
    )
    p_query.add_argument(
        "--model_name",
        type=str,
        # default="microsoft/phi-2",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Generation model name",
    )
    p_query.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name",
    )
    group = p_query.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, help="Single query string")
    group.add_argument("--queries_file", type=str, help="File with one query per line")
    p_query.add_argument("--max_new_tokens", type=int, default=150)
    p_query.add_argument("--temperature", type=float, default=0.7)
    p_query.add_argument("--top_p", type=float, default=0.9)
    p_query.add_argument("--top_k", type=int, default=50)
    p_query.add_argument(
        "--retrieval_top_k", type=int, default=3, help="Number of chunks to retrieve"
    )
    p_query.add_argument(
        "--no_rag", action="store_true", help="Generate without RAG (baseline)"
    )
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        ingest_documents(
            input_file=args.input_file,
            vector_db_path=args.vector_db_path,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    elif args.command == "query":
        if args.query is not None:
            queries = [args.query]
        else:
            queries = read_lines(args.queries_file)
            if not queries:
                raise ValueError("No non-empty lines found in queries file.")
        
        for query in queries:
            answer, context_prompt = generate_with_rag(
                query=query,
                vector_db_path=args.vector_db_path,
                model_name=args.model_name,
                embedding_model_name=args.embedding_model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                retrieval_top_k=args.retrieval_top_k,
                use_rag=not args.no_rag,
            )
            
            print("=" * 80)
            print("QUERY:", query)
            print("=" * 80)
            if context_prompt:
                print("\n[Context used in generation]")
                print(context_prompt)
                print("\n" + "-" * 80)
            print("\nANSWER:", answer)
            print("=" * 80)
            print()


if __name__ == "__main__":
    main()
