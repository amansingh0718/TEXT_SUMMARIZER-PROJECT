import streamlit as st
import torch
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -----------------------
# Load model only once
# -----------------------
@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")
    model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_model()


# -----------------------
# Text cleaning function
# -----------------------
def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text


# -----------------------
# Summarization function
# -----------------------
def summarize_dialogue(dialogue):

    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(
    page_title="Dialogue Summarizer",
    page_icon="📝",
    layout="centered"
)

st.title("📝 Dialogue Summarization using T5")
st.write("Enter a conversation below and click **Generate Summary**.")

dialogue = st.text_area(
    "Input Dialogue",
    height=300,
    placeholder="Paste your dialogue here..."
)

if st.button("Generate Summary"):

    if dialogue.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarize_dialogue(dialogue)

        st.subheader("Summary")
        st.success(summary)