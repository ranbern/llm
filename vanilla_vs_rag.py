import os
import faiss
import numpy as np
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer
# Removed AutoModelForCausalLM and torch as they are not directly used with llama-cpp-python for model loading
from transformers import AutoTokenizer # Still needed for consistent [BLANK_AUDIO] tokenization and chat templates
from llama_cpp import Llama # Import Llama from llama_cpp

# --- Configuration ---
RAG_DATA_DIR = "rag_indexed_corpus_tilman"
INDEX_FILENAME = "faiss_index"

DATASET_PATH = RAG_DATA_DIR
INDEX_PATH = os.path.join(RAG_DATA_DIR, INDEX_FILENAME)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Define the Base LLM to use for both Vanilla and RAG generation
# IMPORTANT: This is now the tokenizer name. The actual model weights will be a GGUF file.
#BASE_LLM_MODEL_NAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
# Assuming your variable for the tokenizer is TOKENIZER_MODEL_NAME
TOKENIZER_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2" # Correct Hugging Face model ID for the tokenizer
GGUF_MODEL_PATH = "mistral-7b-instruct-v0.2.Q4_K_M.gguf" # This is correct for the LLM

# RAG Query Config
K_RAG_RETRIEVE = 5

COMPARISON_QUERY = "which obstacle is related to the water elemnt? Provide one answer out of the following: 1. Ignorance, 2. Pride, 3. Anger, 4. Fear, 5. Desire, 6. Attachment, 7. Jealousy"

# --- Helper Functions ---

def load_llm_model(tokenizer_model_name: str, gguf_model_path: str):
    """
    Loads the LLM model using llama-cpp-python (for GGUF) and the tokenizer.
    """
    print(f"Loading tokenizer for model: {tokenizer_model_name}...")
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        # Add padding token if missing, which is common for causal models
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
        print("✅ Tokenizer loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading tokenizer {tokenizer_model_name}: {e}")
        exit() # Cannot proceed without tokenizer

    print(f"Loading GGUF LLM model from: {gguf_model_path}...")
    try:
        llm_model = Llama(
            model_path=gguf_model_path,
            n_gpu_layers=0,  # Set to 0 to force CPU-only inference
            n_ctx=4096,      # Context window size. Adjust based on your needs.
            n_threads=None   # Number of CPU threads to use. None uses all available.
        )
        print("✅ LLM model loaded successfully on CPU.")
    except Exception as e:
        print(f"❌ Error loading GGUF model from {gguf_model_path}: {e}")
        print("Please ensure the GGUF file exists at the specified path and llama-cpp-python is correctly installed.")
        exit() # Cannot proceed without the LLM

    return llm_model, llm_tokenizer

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

# --- Query Functions ---

def generate_llm_response(prompt_text: str, llm_model: Llama, llm_tokenizer: AutoTokenizer):
    """Helper to generate response using llama_cpp.Llama model."""
    # Mixtral Instruct uses a specific chat template for best performance
    messages = [{"role": "user", "content": prompt_text}]
    formatted_prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    output = llm_model.create_completion(
        formatted_prompt,
        max_tokens=250, # Max new tokens for the answer
        temperature=0.1, # Lower for more focused answers
        top_p=0.9,
        stop=["<|im_end|>", "</s>"] # Common stop tokens for Mixtral/Llama
    )
    # The output from create_completion is the generated text after the prompt
    response_text = output["choices"][0]["text"]
    return response_text.strip()

def vanilla_query(query: str, llm_model: Llama, llm_tokenizer: AutoTokenizer):
    """Queries the base LLM directly without external context."""
    print(f"\n--- Running Vanilla LLM Query ---")
    print(f"Query: '{query}'")

    vanilla_prompt = f"Answer the following question:\n{query}\n\nAnswer:"

    try:
        generated_answer = generate_llm_response(vanilla_prompt, llm_model, llm_tokenizer)
        print("✅ Answer generated.")
        return generated_answer
    except Exception as e:
        print(f"❌ Error during Vanilla LLM generation: {e}")
        return f"Error during vanilla answer generation: {e}"

def rag_query(query: str, dataset: Dataset, faiss_index: faiss.Index, embedding_model: SentenceTransformer, llm_model: Llama, llm_tokenizer: AutoTokenizer, k: int = K_RAG_RETRIEVE):
    """
    Performs RAG retrieval using loaded components and generates an answer with the LLM.
    Uses the raw faiss_index object for search.
    """
    print(f"\n--- Running RAG Query ---")
    print(f"Query: '{query}'")
    print(f"Retrieving top {k} documents from corpus...")

    query_embedding = embedding_model.encode(query, convert_to_numpy=True).astype(np.float32).reshape(1, -1)

    try:
        distances, retrieved_examples_indices = faiss_index.search(query_embedding, k=k)
        retrieved_indices = retrieved_examples_indices[0]
        retrieved_scores = distances[0]
    except Exception as e:
         print(f"❌ Error during FAISS search: {e}")
         return f"Error during RAG retrieval: {e}"

    print("\n--- Retrieved Documents ---")
    retrieved_context = ""
    valid_indices = [idx.item() for idx in retrieved_indices if idx.item() >= 0 and idx.item() < len(dataset)]

    if not valid_indices:
         print("❌ No valid documents retrieved.")
         return "No relevant information found in the corpus."

    for i, doc_index in enumerate(valid_indices):
        original_rank_index = list(retrieved_examples_indices[0]).index(doc_index)
        score = retrieved_scores[original_rank_index].item()

        try:
            document = dataset[doc_index]
            print(f"--- Retrieved Doc Index: {doc_index} ---")
            print(f"Title: {document.get('title', 'N/A')}")
            print(f"Text (Excerpt): {document.get('text', 'N/A')[:200]}...") # Print excerpt for brevity
            print(f"Score: {score:.4f}")
            print("---------------------------------------")
            retrieved_context += f"Title: {document.get('title', 'N/A')}\nText: {document.get('text', 'N/A')}\n\n"
        except Exception as e:
             print(f"Warning: Could not retrieve document at index {doc_index} from dataset: {e}")

    if not retrieved_context.strip():
        print("❌ Retrieved documents were empty or could not be processed.")
        return "Retrieved content was empty."

    print("\n--- Generating Answer with LLM and Context ---")
    rag_prompt = f"""Using the following context, answer the user's question.
If you cannot answer the question based on the context, say "I cannot answer based on the provided information."

Context:
{retrieved_context.strip()}

Question: {query}

Answer:
"""
    try:
        generated_answer = generate_llm_response(rag_prompt, llm_model, llm_tokenizer)
        print("✅ Answer generated.")
        return generated_answer
    except Exception as e:
        print(f"❌ Error during LLM generation (with RAG context): {e}")
        return f"Error during RAG answer generation: {e}"

# --- Main Comparison Execution ---

if __name__ == "__main__":
    print("--- Loading Saved RAG Data ---")
    print(f"Loading dataset from {DATASET_PATH}...")
    try:
        dataset = load_from_disk(DATASET_PATH)
        print(f"✅ Dataset loaded successfully with {len(dataset)} rows.")
        if 'embeddings' not in dataset.column_names:
             print("Warning: 'embeddings' column not found in the loaded dataset. This is unexpected if it was saved correctly.")
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

    print("\n--- Loading Models ---")
    # Call load_llm_model with both tokenizer name and GGUF path
    llm_model, llm_tokenizer = load_llm_model(TOKENIZER_MODEL_NAME, GGUF_MODEL_PATH)

    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    if embedding_model is None:
        print("Cannot proceed without embedding model.")
        exit()

    print(f"\n--- Running Comparison Query: '{COMPARISON_QUERY}' ---")

    vanilla_answer = vanilla_query(COMPARISON_QUERY, llm_model, llm_tokenizer)
    rag_answer = rag_query(COMPARISON_QUERY, dataset, faiss_index, embedding_model, llm_model, llm_tokenizer, k=K_RAG_RETRIEVE)

    print("\n--- Comparison Results ---")
    print(f"Query: {COMPARISON_QUERY}")
    print("\n--- Vanilla LLM Answer ---")
    print(vanilla_answer)
    print("\n--- RAG LLM Answer ---")
    print(rag_answer)
    print("\n--------------------------")

    print("\nComparison Complete.")

    del llm_model
    del llm_tokenizer
    del embedding_model
    del faiss_index