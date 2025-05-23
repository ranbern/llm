import os
import faiss
import numpy as np
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer

# --- Configuration (MUST MATCH YOUR SAVED RAG DATA) ---
RAG_DATA_DIR = "rag_indexed_corpus_tilman" # <--- Ensure this is the correct directory
INDEX_FILENAME = "faiss_index"            # The name used when saving the index file

DATASET_PATH = RAG_DATA_DIR
INDEX_PATH = os.path.join(RAG_DATA_DIR, INDEX_FILENAME)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # <--- Must be the same model used for embedding the corpus

# Retrieval Config
K_RETRIEVE = 5 # Number of documents to retrieve for the query

# --- Helper Functions (from your existing code) ---

def load_embedding_model(model_name: str):
    """Loads the SentenceTransformer embedding model."""
    print(f"Loading embedding model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        print("✅ Embedding model loaded.")
        return model
    except Exception as e:
         print(f"❌ Error loading embedding model: {e}")
         print(f"Please ensure you have an internet connection or the model '{model_name}' is cached, and the sentence-transformers library is installed (`pip install sentence-transformers`).")
         return None

# --- Main Retrieval Logic ---

def get_and_print_relevant_chunks(
    query: str,
    dataset: Dataset,
    faiss_index: faiss.Index,
    embedding_model: SentenceTransformer,
    k: int = K_RETRIEVE
):
    """
    Given a query, retrieves the top K relevant chunks from the FAISS index
    and prints their title, text, and similarity score.
    """
    print(f"\n--- Retrieving Top {k} Chunks for Query: '{query}' ---")

    # Embed query
    query_embedding = embedding_model.encode(query, convert_to_numpy=True).astype(np.float32)
    query_embedding = query_embedding.reshape(1, -1) # Reshape for FAISS search

    # Search FAISS index
    try:
        distances, retrieved_examples_indices = faiss_index.search(query_embedding, k=k)

        # Get indices and scores for the first (and only) query
        retrieved_indices = retrieved_examples_indices[0]
        retrieved_scores = distances[0] # Note: FAISS returns distance, lower is better for L2, higher for IP (if used)

    except Exception as e:
        print(f"❌ Error during FAISS search: {e}")
        return

    print("\n--- Retrieved Chunks ---")
    retrieved_chunk_details = [] # To store details before printing, for cleaner output

    # Filter out invalid indices and gather details
    valid_results = []
    for i, idx in enumerate(retrieved_indices):
        if idx >= 0 and idx < len(dataset): # Ensure index is valid
            valid_results.append((idx.item(), retrieved_scores[i].item()))
        else:
            print(f"Warning: Invalid index {idx.item()} returned by FAISS at rank {i+1}. Skipping.")

    if not valid_results:
        print("No relevant chunks found.")
        return

    # Now iterate through valid results and print
    for i, (doc_index, score) in enumerate(valid_results):
        try:
            document = dataset[doc_index]
            title = document.get('title', 'N/A')
            text = document.get('text', 'N/A')

            retrieved_chunk_details.append(
                f"--- Chunk {i+1} ---"
                f"\nRank: {i+1}"
                f"\nScore: {score:.4f} (Lower score means more similar for L2 distance, higher for IP similarity)"
                f"\nOriginal Dataset Index: {doc_index}"
                f"\nTitle: {title}"
                f"\nText:\n{text}\n"
                f"{'-' * 40}\n" # Separator
            )
        except Exception as e:
            print(f"Warning: Could not retrieve document at index {doc_index} from dataset: {e}")

    # Print all collected chunk details
    for detail in retrieved_chunk_details:
        print(detail)


# --- Main Execution ---

if __name__ == "__main__":
    # --- Step 1: Load the saved RAG components (Dataset and FAISS Index) ---
    print("--- Loading Saved RAG Data ---")
    print(f"Loading dataset from {DATASET_PATH}...")
    try:
        dataset = load_from_disk(DATASET_PATH)
        print(f"✅ Dataset loaded successfully with {len(dataset)} rows.")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"Please ensure '{DATASET_PATH}' is the correct path to your saved dataset.")
        exit()

    print(f"Loading FAISS index from {INDEX_PATH}...")
    try:
        faiss_index = faiss.read_index(INDEX_PATH)
        print(f"✅ FAISS index loaded successfully with {faiss_index.ntotal} vectors.")
    except Exception as e:
        print(f"❌ Error loading FAISS index: {e}")
        print(f"Please ensure '{INDEX_PATH}' is the correct path to your saved index file.")
        exit()

    # --- Step 2: Load the Embedding Model ---
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    if embedding_model is None:
        print("Cannot proceed without embedding model.")
        exit()

    # --- Step 3: Get Query from User and Print Chunks ---
    while True:
        user_query = input("\nEnter your query (or 'exit' to quit): ").strip()
        if user_query.lower() == 'exit':
            break

        get_and_print_relevant_chunks(user_query, dataset, faiss_index, embedding_model, k=K_RETRIEVE)

    print("\nExiting chunk retrieval.")

    # Clean up loaded objects (optional)
    del embedding_model
    del faiss_index
    # del dataset # Dataset object is typically small enough that explicit deletion isn't critical