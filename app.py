import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Page Configuration
st.set_page_config(page_title="Toxic Comment Classifier", page_icon="🛡️", layout="centered")

st.title("🛡️ Content Moderation AI")
st.write("Analyze text for toxic patterns using a fine-tuned NLP Transformer.")

# 2. Cache the model so it only loads once per session
@st.cache_resource
def load_model():
    # Points to your newly extracted folder
    model_path = "./toxic_comment_model" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

with st.spinner("Loading AI Model... (This takes a few seconds on the first run)"):
    tokenizer, model = load_model()

# 3. User Input
comment = st.text_area("Enter a comment to analyze:", height=150, placeholder="Type something here...")

# 4. Prediction Logic
if st.button("Analyze Comment", type="primary"):
    if not comment.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            # Tokenize the input
            inputs = tokenizer(comment, return_tensors="pt", padding=True, truncation=True, max_length=128)
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Calculate percentages
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            toxic_score = probabilities[0][1].item()
            
            # Display Results
            st.divider()
            if toxic_score > 0.5:
                st.error(f"🚨 **Toxic Content Detected**")
                st.write(f"Confidence: **{toxic_score * 100:.2f}%**")
                st.progress(toxic_score)
            else:
                st.success(f"✅ **Content Appears Safe**")
                st.write(f"Toxicity Probability: **{toxic_score * 100:.2f}%**")
                st.progress(toxic_score)