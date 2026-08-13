import os
import glob

# Importing sentence Transformers
try:
    from sentence_transformers import SentenceTransformer

except Exception as e:
    print(f"[Fatal Error] Could not import sentence-transformers: {e}")
    raise

# Importing ChromaDB
try:
    import chromadb

except Exception as e:
    print(f"[Fatal Error] Could not import chromabd: {e}")
    raise

# Configuration values
docs_dir = "docs"
chroma_dir = "chromadb"
collection_name = "Zepto_policy_corpus"
embedding_model_name = "all-MiniLM-L6-v2"

# Loading documents
def load_documents(docs_dir: str) -> dict:
    """Load every .txt files in the docs folder i.e., docs_dir into a {doc_id: text} which expected to return dictionary (dict)"""
    # Create an empty dictionary
    documents = {}

    try:
        # Finding file paths
        file_paths = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))

        # Checking whether files exists
        if not file_paths:
            raise FileNotFoundError(f"No. of text files found in '{docs_dir}'. Expected doc_01.txt through doc_08.txt")

        # Extracting document ID
        for path in file_paths:
            file_name = os.path.basename(path)
            doc_id = os.path.splitext(file_name)[0]

            try:
                # Reading the document from the folder location
                with open(path, "r", encoding= "utf-8") as f:
                    documents[doc_id] = f.read().strip()

            except Exception as e:
                print(f"[Error] Failed to read '{path}: {e}")

        print(f"Loaded {len(documents)} documents from '{docs_dir}'.")

    except Exception as e:
        print(f"[Error] load_documnets failed: {e}")

    return documents

# Perform Chunking
def chunk_documents(documents: dict) -> dict:
    """
    Simple per-document chunking: each document is short, so one chunk per document is enough. 
    Returns {chunk_id: chunk_text} -- chunk_id == doc_id since there is exactly one chunk per document.
    """
    # Create a chunk dictionary
    chunks = {}

    try:
        # Copy the whole document as one chunk
        for doc_id, text in documents.items():
            chunks[doc_id] = text
        print(f"Created {len(chunks)} chunks (one per document)")

    except Exception as e:
        print(f"[Error] chunk_documents failed: {e}")

    return chunks

# Create embeddings and store them in chomadb
def embeddings_and_store(chunks: dict):
    """Convert small pieces of text (chunks) into numerical arrays (embeddings) using local computer model,
    and then save/update those arrays (embeddings) inside a local database (ChromaDB) so we can search those embeddings later"""

    try:
        # Loading the embedding model
        print(f"Loading embedding model '{embedding_model_name}'")
        model = SentenceTransformer(embedding_model_name)
    
    except Exception as e:
        print(f"[Fatal Error] Could not load embedding model: {e}")
        return None

    # Creating the ChromaDB client
    try:
        client = chromadb.PersistentClient(path= chroma_dir)
        # create the collection
        collection = client.get_or_create_collection(name= collection_name)

    except Exception as e:
        print(f"[Fatal Error] Could not initialize ChromaDB collection: {e}")
        return None

    # Extracting Chunk_ids and chunk_text
    try:
        chunk_ids = list(chunks.keys())
        chunk_texts = list(chunks.values())

        # Creating embeddings
        embeddings = model.encode(chunk_texts).tolist()

        # Storing everything in ChromaDB
        collection.upsert(ids= chunk_ids, documents= chunk_texts, embeddings= embeddings)
        print(f"Stored {len(chunk_ids)} embedded chunks in ChromaDB collection '{collection_name}' at '{chroma_dir}'")
        return collection

    except Exception as e:
        print(f"[Error] embeddings_and_store failed while upserting: {e}")
        return None

# Now create main function
def main():
    try:
        # Load the docs
        documents = load_documents(docs_dir)
        if not documents:
            print(f"[Fatal Error] No documents loaded. \n-----aborting chunking and embeddings")
            return

        # Perform Chunking
        chunks = chunk_documents(documents)
        if not chunks:
            print(f"[Fatal Error] No chunks created. \n-----aborting chunking and embeddings")
            return

        # Perform embeddings and storing
        collection = embeddings_and_store(chunks)
        # Checking the final count
        if collection is not None:
            try:
                count = collection.count()
                print(f"Chunking and embeddings completed. Collection now contains {count} embedded chunks.")

            except Exception as e:
                print(f"[Error] Could not confirm final collection count: {e}")

        else:
            print(f"[Fatal Error] Chunking and embeddings did not complete successfully.")

    except Exception as e:
        print(f"[Fatal Error] Unexpected error in ingest.py: {e}")


if __name__ == "__main__":
    main()