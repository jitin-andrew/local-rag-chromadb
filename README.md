work nature:
document ingestion: splits documents into chunks, generates embeddings using sentence-transformers/all-MiniLM-L6-v2 and stores them in a local chromaDB vector database.
query processing: for each query, retrieves the most relevant document chunks using cosine similarity and then uses a local lm (TinyLlama-1.1B-Chat) to generate answers grounded in the retrieved context.

features:
completely local - no external services required.
efficient for small documents - optimized for minimal text documents
context-aware answers — retrieves relevant facts before generating responses
lightweight models — uses small, efficient models suitable for local deployment

uses:
great for querying company documentation, fact sheets or any small collection of text documents where you need accurate, context-based answers without fine-tuning a model or external services.

steps:
1) ingest your docs:

```bash
python3 finetune_generate.py ingest \
  --input_file sample.txt \
  --vector_db_path ./chroma_db
```
2) generate:

```bash
python finetune_generate.py query \
  --query "what services does [x] company provide?" \
  --vector_db_path ./chroma_db
```
