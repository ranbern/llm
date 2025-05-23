import os
import streamlit as st
import psutil
st.write(f"Available RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")

st.title("Language Model Comparison")
st.write("Compare outputs from vanilla and fine-tuned models.")

# Input area
query = st.text_area("Enter your prompt:", value="Once upon a time", height=100)

# Simplified options - just one model at a time
model_choice = st.selectbox(
    "Select model to run:",
    ["Vanilla (distilgpt2)", "Fine-tuned model"]
)

# Parameter for generation length
new_tokens = st.slider("Number of tokens to generate:", 10, 150, 50)

# Generate button
if st.button("Generate Text"):
    # Check if fine-tuned model exists if that's what was selected
    if model_choice == "Fine-tuned model" and not os.path.exists("./fine_tuned_model"):
        st.error("Fine-tuned model not found. Please run training first.")
    else:
        with st.spinner("Setting up environment..."):
            # Import libraries only when needed
            import torch
            import os

            # Force CPU usage and disable CUDA
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            torch.set_num_threads(1)  # Limit CPU threads

        # Choose model path based on selection
        model_path = "./fine_tuned_model" if model_choice == "Fine-tuned model" else "distilgpt2"

        with st.spinner(f"Loading {model_choice}..."):
            # Import transformers here to avoid early initialization issues
            from transformers import AutoTokenizer, AutoModelForCausalLM

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="cpu"  # Remove `low_cpu_mem_usage=True`
            )

        with st.spinner("Generating text..."):
            # Basic tokenization - keep it short
            inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=128)

            # Generate with simple settings
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=new_tokens,
                do_sample=False,  # For testing and predictability
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # Decode the generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean up memory
            del model, tokenizer, inputs, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            import gc

            gc.collect()

        # Display the results
        st.success("Text generation complete!")
        st.subheader("Generated text:")
        st.write(generated_text)