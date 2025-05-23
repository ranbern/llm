import os
import json
import faiss
import numpy as np
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer
import torch

# --- Configuration ---
CORPUS_FILE = "tilman.json"
RAG_DATA_DIR = "rag_indexed_corpus_tilman"
INDEX_FILENAME = "faiss_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Chunking Parameters ---
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100

# --- Helper Functions ---

def load_corpus(file_path: str):
    """
    Loads the corpus from a JSON file and splits transcriptions into fixed-size chunks with overlap.
    Each chunk will become a separate entry in the dataset.
    """
    print(f"Loading corpus from JSON file: {file_path} and chunking text into {CHUNK_SIZE}-char chunks with {CHUNK_OVERLAP}-char overlap...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)

        chunked_corpus_list = []
        if isinstance(corpus_data, dict):
            for title, full_text in corpus_data.items():
                start_index = 0
                chunk_id_counter = 0
                while start_index < len(full_text):
                    end_index = start_index + CHUNK_SIZE
                    chunk_text = full_text[start_index:end_index].strip()

                    if chunk_text: # Only add non-empty chunks
                        chunk_title = f"{title}_chunk_{chunk_id_counter}"
                        chunked_corpus_list.append({"title": chunk_title, "text": chunk_text})
                        chunk_id_counter += 1

                    # Move the start index for the next chunk, accounting for overlap
                    start_index += (CHUNK_SIZE - CHUNK_OVERLAP)
                    # Ensure start_index doesn't go backward if chunk_overlap is larger than chunk_size (though it shouldn't be)
                    if start_index < 0:
                        start_index = 0
        else:
            print("❌ Error: Expected corpus JSON to be a dictionary of 'filename': 'transcription_text'.")
            return None

        corpus = Dataset.from_list(chunked_corpus_list)
        print(f"✅ Loaded {len(corpus)} chunks from {file_path}.")
        return corpus
    except FileNotFoundError:
        print(f"❌ Error: Corpus file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in {file_path}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred while loading corpus: {e}")
        return None

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

def embed_corpus(corpus: Dataset, embedding_model: SentenceTransformer):
    """Generates embeddings for the corpus."""
    print("Generating embeddings for the corpus...")
    # Get the texts from the 'text' column of your dataset
    texts = corpus['text']
    # Generate embeddings
    embeddings = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # Add embeddings as a new column to the dataset
    corpus = corpus.add_column("embeddings", [emb for emb in embeddings])
    print("✅ Embeddings generated and added to dataset.")
    return corpus

def create_and_save_faiss_index(dataset_with_embeddings: Dataset, save_dir: str, index_filename: str):
    """Creates a FAISS index from the embeddings and saves it along with the dataset."""
    print("Creating FAISS index...")
    # FAISS expects numpy array of float32
    embeddings = np.array(dataset_with_embeddings["embeddings"]).astype('float32')
    embedding_dim = embeddings.shape[1]

    # Use an IndexFlatL2 for simple similarity search (Euclidean distance)
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(embeddings) # Add embeddings to the index

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the FAISS index
    faiss.write_index(faiss_index, os.path.join(save_dir, index_filename))
    print(f"✅ FAISS index saved to {os.path.join(save_dir, index_filename)}")

    # Save the dataset (which now includes embeddings)
    dataset_with_embeddings.save_to_disk(save_dir)
    print(f"✅ Dataset with embeddings saved to {save_dir}")
    return faiss_index

def load_rag_components(data_dir: str, index_filename: str):
    """Loads the saved dataset and FAISS index."""
    dataset_path = data_dir
    index_path = os.path.join(data_dir, index_filename)

    print(f"Loading dataset from {dataset_path}...")
    try:
        dataset = load_from_disk(dataset_path)
        print(f"✅ Dataset loaded successfully with {len(dataset)} rows.")
    except Exception as e:
        print(f"❌ Error loading dataset from {dataset_path}: {e}")
        print(f"Please ensure the directory '{dataset_path}' exists and contains the saved dataset files.")
        return None, None

    print(f"Loading FAISS index from {index_path}...")
    try:
        faiss_index = faiss.read_index(index_path)
        print(f"✅ FAISS index loaded successfully with {faiss_index.ntotal} vectors.")
    except Exception as e:
        print(f"❌ Error loading FAISS index from {index_path}: {e}")
        print(f"Please ensure '{index_path}' is the correct path to your saved index file.")
        return None, None

    return dataset, faiss_index


# --- Main Execution ---

if __name__ == "__main__":
    if os.path.exists(RAG_DATA_DIR) and \
       os.path.exists(os.path.join(RAG_DATA_DIR, INDEX_FILENAME)):
        print(f"\nFound existing RAG components in {RAG_DATA_DIR}. Loading...")
        dataset, faiss_index = load_rag_components(RAG_DATA_DIR, INDEX_FILENAME)

        if dataset is None or faiss_index is None:
            print("Failed to load existing RAG components. Please check the error messages above.")
            exit() # Exit if loading failed

    else:
        print("\nNo existing RAG components found. Preparing RAG data...")
        # 1. Load Corpus
        corpus = load_corpus(CORPUS_FILE)
        if corpus is None:
            exit()

        # 2. Load Embedding Model
        embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
        if embedding_model is None:
            exit()

        # 3. Embed Corpus
        dataset_with_embeddings = embed_corpus(corpus, embedding_model)

        # 4. Create and Save FAISS Index along with the Dataset
        faiss_index = create_and_save_faiss_index(dataset_with_embeddings, RAG_DATA_DIR, INDEX_FILENAME)

        # If we just created them, these are our components for potential further use
        dataset = dataset_with_embeddings

    print("\nRAG data preparation complete.")
    # At this point, 'dataset' and 'faiss_index' are ready to be used
    # For example, you could then proceed to run your RAG queries or comparisons
    print("Dataset and FAISS index are ready for use.")