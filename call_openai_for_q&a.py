# === Setup ===
import os
import json
import re # Import regex for more robust sentence splitting
from openai import OpenAI
from tabulate import tabulate

# Set up OpenAI API (ensure your API key is set as an environment variable or replace this line)
os.environ["OPENAI_API_KEY"] = "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"


client = OpenAI()

# === Utility: Token limits by model ===
def get_token_limit_for_model(model_name):
    """
    Returns the maximum context window size for a given OpenAI model.
    """
    model_context_limits = {
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
    }
    return model_context_limits.get(model_name, 4096)

def get_chunking_params(model_name):
    """
    Calculates parameters for text chunking and API call limits.
    Adjusts input chunk size and output token limit to prevent truncation.
    """
    token_limit = get_token_limit_for_model(model_name)

    # Max tokens for the actual transcript content in a single input chunk.
    # This is set to a smaller percentage to ensure output space and prevent truncation.
    # Tuned to 20% of the model's total context window.
    max_input_for_chunk_content_tokens = int(token_limit * 0.20)

    # Rough estimate of words per chunk. Assuming ~0.75 words per token for English.
    words_per_chunk = int(max_input_for_chunk_content_tokens * 0.75)

    # Explicitly limit the generated response length (output tokens).
    # This is crucial to prevent the JSON from being truncated mid-object.
    # Increased to 1000 to allow for more complete Q&A pairs.
    max_output_tokens = 3000 # This value might need further tuning based on Q&A verbosity

    print(f"DEBUG: token_limit={token_limit}, max_input_for_chunk_content_tokens={max_input_for_chunk_content_tokens}")
    print(f"DEBUG: words_per_chunk={words_per_chunk}, max_output_tokens={max_output_tokens}")

    return max_input_for_chunk_content_tokens, words_per_chunk, max_output_tokens

# === Prompts ===
SYSTEM_PROMPT = """You are a Dharma teacher and language model trainer.

Your task is to process a full transcript of a Dharma teaching and extract a **comprehensive set of deep, meaningful, and highly granular Q&A pairs**. These pairs are for fine-tuning a language model, so **overlaps, similar questions, and some duplication of content are acceptable and even desired** to ensure thorough coverage.

Each pair must include:
- A broad and meaningful question reflecting a specific point or theme in the talk.
- A representative quote from the transcript (a few sentences or a short paragraph).
- A rephrased answer that clearly explains the idea for a general audience.

**Systematically go through the transcript and extract Q&A pairs for every distinct idea, concept, or teaching point presented.** Do not limit yourself to only major topics; capture the nuances.

Output a JSON list of objects with this schema:
[
  {
    "source": "<source>",
    "question": "...",
    "answer_quote": "...",
    "answer_rephrased": "..."
  }
]
"""

USER_PROMPT_TEMPLATE = """Below is the full transcript of a Dharma talk from source {source}.
Please semantically chunk it into key sections, and for each section create a meaningful Q&A pair.

Transcript:
\"\"\"
{text}
\"\"\""""

# === API Call ===
def generate_semantic_qas(text, source, model, max_output_tokens, retry_level=0, max_retries=2):
    """
    Calls the OpenAI API to generate semantic Q&A pairs from a given text chunk.
    Includes robust JSON parsing and a retry mechanism for malformed responses.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(source=source, text=text)
    content = "" # Initialize content to handle potential errors before response.choices
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=max_output_tokens # Limit the output tokens to prevent truncation
        )

        content = response.choices[0].message.content
        json_string = ""
        data = []

        # Use regex to find the JSON array within the content, handling multi-line content.
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            json_string = match.group(0)
            try:
                data = json.loads(json_string)
                # If successful, return the data
                return data
            except json.JSONDecodeError as e:
                # If full parse fails, attempt to repair by finding the last complete object
                print(f"⚠️ JSONDecodeError after full match for {source} at retry_level {retry_level}: {e}")
                print(f"Problematic string segment (original):\n{json_string}")
                last_brace_idx = json_string.rfind('}')
                if last_brace_idx != -1:
                    repaired_json_string = json_string[:last_brace_idx + 1] + "]"
                    try:
                        data = json.loads(repaired_json_string)
                        print(f"✅ Successfully parsed partial JSON for {source} at retry_level {retry_level}")
                        return data # Return partially parsed data
                    except json.JSONDecodeError as e_repaired:
                        print(f"❌ Failed to parse repaired JSON for {source} at retry_level {retry_level}: {e_repaired}")
                        print("Problematic string segment (repaired):\n", repaired_json_string)
                        # Fall through to retry logic if repair fails
                else:
                    print(f"❌ No complete JSON objects found in response for {source} at retry_level {retry_level}")
                    # Fall through to retry logic if no complete objects
        else:
            print(f"❌ Could not find JSON array in response for {source} at retry_level {retry_level}")
            print("Raw response:\n", content)
            print('Problematic chunk:', text)
            # Fall through to retry logic if no JSON array found

        # --- Retry Logic ---
        if retry_level < max_retries:
            print(f"Retrying {source} by splitting into smaller chunks (retry_level {retry_level + 1})...")
            # Define a smaller chunk size for retry. This can be tuned.
            # For example, half of the initial words_per_chunk, or a fixed small number like 150-200.
            # Using a fixed small number to ensure very fine-grained splitting on retry.
            retry_chunk_words = 150

            sub_chunks = split_long_text(text, retry_chunk_words)
            retried_qas = []
            for i, sub_chunk in enumerate(sub_chunks):
                sub_source = f"{source}_retry{retry_level+1}c{i+1}"
                # Recursive call with incremented retry_level
                qa = generate_semantic_qas(sub_chunk, sub_source, model, max_output_tokens, retry_level + 1, max_retries)
                retried_qas.extend(qa)
            return retried_qas
        else:
            print(f"❌ Max retries ({max_retries}) reached for {source}. Skipping this chunk due to persistent malformed response.")
            return []

    except Exception as e:
        print(f"An unexpected error occurred during API call for {source} at retry_level {retry_level}: {e}")
        if content: # Only print content if it was successfully retrieved
            print("Raw response:\n", content)
        return []

# === Text splitting ===
def split_long_text(text, max_words):
    """
    Splits a long text into chunks of a maximum number of words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

# === Text Analysis Helpers (Custom implementations without NLTK) ===
def count_words(text):
    """Counts words by splitting on whitespace and filtering empty strings."""
    return len([word for word in text.split() if word])

def count_tokens(text):
    """
    Counts 'tokens' by finding sequences of word characters.
    This is a simplified tokenization.
    """
    return len([token for token in re.findall(r'\b\w+\b', text.lower()) if token])

def count_sentences(text):
    """
    Counts sentences by splitting on common sentence-ending punctuation.
    Handles multiple punctuation marks and trailing whitespace.
    """
    sentences = re.split(r'[.!?…]+(?:\s+|\Z)', text)
    return len([s for s in sentences if s.strip()])

# === Main processor ===
def process_all_transcripts(transcript_file='transcripts.json', output_file='semantic_qa_output.json', model="gpt-4"):
    """
    Main function to process all transcripts, generate Q&A pairs,
    and print summary tables.
    """
    max_input_for_chunk_content_tokens, words_per_chunk, max_output_tokens = get_chunking_params(model)
    print(f"\n🔧 Using model: {model}")
    print(f"   → Max Input Tokens per chunk: {max_input_for_chunk_content_tokens}")
    print(f"   → Word chunk size for input: {words_per_chunk}")
    print(f"   → Max Output Tokens per API call: {max_output_tokens}\n")

    try:
        talk2transcript = json.load(open(transcript_file, 'r'))
    except FileNotFoundError:
        print(f"Error: Transcript file '{transcript_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{transcript_file}'. Please check file format.")
        return

    all_qas = []

    # Dataset summary counters
    total_qas_overall = 0
    total_quote_words_overall = 0
    total_transcript_words_overall = 0

    for source, text in talk2transcript.items():
        source_id = source.split('.')[0]
        print(f"\n--- Processing: {source_id} ---")

        transcript_word_count = count_words(text)
        total_transcript_words_overall += transcript_word_count

        current_transcript_qas = []
        if transcript_word_count <= words_per_chunk:
            print("   ✅ Sending full transcript")
            # Initial call to generate_semantic_qas with retry_level=0
            qa = generate_semantic_qas(text, source_id, model, max_output_tokens, retry_level=0)
            current_transcript_qas.extend(qa)
        else:
            print(f"   ⚠️ Transcript too long ({transcript_word_count} words), splitting...")
            chunks = split_long_text(text, words_per_chunk)
            for i, chunk in enumerate(chunks):
                print(f"   → Sub-chunk {i+1}/{len(chunks)}")
                # Initial call to generate_semantic_qas for each chunk with retry_level=0
                qa = generate_semantic_qas(chunk, f"{source_id}_part{i+1}", model, max_output_tokens, retry_level=0)
                current_transcript_qas.extend(qa)

        all_qas.extend(current_transcript_qas)

        # --- First Table: Q&A Details for Current Transcript ---
        table_1_headers = ["Question (snippet)", "Quote Words", "Quote Tokens", "Quote Sentences", "Quote % of Transcript", "Question Words", "Rephrased Answer Words"]
        table_1_data = []

        current_transcript_total_quote_words = 0

        for qa_pair in current_transcript_qas:
            # --- Robustness check for missing keys ---
            required_keys = ["question", "answer_quote", "answer_rephrased"]
            if not all(key in qa_pair and qa_pair[key] is not None for key in required_keys):
                print(f"⚠️ Skipping malformed Q&A pair for {source_id}: Missing or None value for one or more required keys.")
                print(f"Malformed pair: {qa_pair}")
                continue # Skip this pair and move to the next one

            quote_words = count_words(qa_pair["answer_quote"])
            quote_tokens = count_tokens(qa_pair["answer_quote"])
            quote_sentences = count_sentences(qa_pair["answer_quote"])

            # Calculate percentage of whole number of words in the transcript
            quote_percentage = (quote_words / transcript_word_count * 100) if transcript_word_count > 0 else 0

            question_words = count_words(qa_pair["question"])
            rephrased_answer_words = count_words(qa_pair["answer_rephrased"])

            table_1_data.append([
                qa_pair["question"][:50] + "..." if len(qa_pair["question"]) > 50 else qa_pair["question"],
                quote_words,
                quote_tokens,
                quote_sentences,
                f"{quote_percentage:.2f}%",
                question_words,
                rephrased_answer_words
            ])
            current_transcript_total_quote_words += quote_words

        print("\n--- Q&A Details for Current Transcript ---")
        if table_1_data: # Only print if there's data to show
            print(tabulate(table_1_data, headers=table_1_headers, tablefmt="grid"))
        else:
            print("No complete Q&A pairs found for this transcript.")

        # --- Second Table: Transcript Summary ---
        num_questions = len(current_transcript_qas)
        total_coverage_by_quotes = (current_transcript_total_quote_words / transcript_word_count * 100) if transcript_word_count > 0 else 0

        table_2_headers = ["Metric", "Value"]
        table_2_data = [
            ["Number of Questions", num_questions],
            ["Total Percentage of Coverage by Quotes", f"{total_coverage_by_quotes:.2f}%"]
        ]
        print("\n--- Transcript Summary ---")
        print(tabulate(table_2_data, headers=table_2_headers, tablefmt="grid"))

        # Update overall counters
        total_qas_overall += num_questions
        total_quote_words_overall += current_transcript_total_quote_words

    # --- Third Table: Dataset Summary (Overall) ---
    overall_coverage_of_quotes = (total_quote_words_overall / total_transcript_words_overall * 100) if total_transcript_words_overall > 0 else 0

    table_3_headers = ["Metric", "Value"]
    table_3_data = [
        ["Total Q&As in Dataset", total_qas_overall],
        ["Total Words in All Quotes", total_quote_words_overall],
        ["Overall Coverage of Quotes out of Whole Corpus", f"{overall_coverage_of_quotes:.2f}%"]
    ]
    print("\n--- Dataset Summary (Overall) ---")
    print(tabulate(table_3_data, headers=table_3_headers, tablefmt="grid"))

    # Save all Q&A pairs to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"conversations": all_qas}, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Saved {len(all_qas)} Q&A pairs to {output_file}")

# === Entry point ===
if __name__ == "__main__":
    process_all_transcripts()
