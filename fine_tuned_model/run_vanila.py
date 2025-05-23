import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cpu")

st.title("DistilGPT2 Local Generator (Transformers)")
st.write("Enter a query and get a local response using DistilGPT2 via HuggingFace Transformers.")


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    # Add a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return tokenizer, model


tokenizer, model = load_model()
model.to(device)
query = st.text_area("Enter your query:", value="Once upon a time", height=150)

if st.button("Generate Response"):
    with st.spinner("Generating response..."):
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        # Move inputs to the correct device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,
            pad_token_id=tokenizer.pad_token_id  # Use the correct pad_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.subheader("Generated Response")
    st.text_area("Response", value=response, height=200)