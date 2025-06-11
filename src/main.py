import streamlit as st
import numpy as np
import fitz
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_preparation import prepare_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "mlp.keras")
model = load_model(model_path)


data = prepare_data()
tokenizer = data["tokenizer"]
MAX_LEN = 300

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded


def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    else:
        return ""

st.title("TEXT SCORER")

uploaded_file = st.file_uploader("Choose a file(.txt or .pdf)", type=["txt", "pdf"])

if uploaded_file is not None:
    text = extract_text(uploaded_file)
    
    if text.strip() == "":
        st.error("Failed to read a file content.")
    else:
        st.success("Successfully read the text!")
        st.text_area("Your text:", text[:1000], height=200)

        if st.button("Analyze with Text Scorer"):
            X = preprocess_text(text)

            pred = model.predict(X)

            st.subheader("📊 Score:")
            st.write("📘 ASAP (grade):", float(pred[0][0]))
            st.write("📗 CommonLit (readability):", float(pred[1][0]))
            st.write("📕 JFLEG (is text grammary correct):", "YES" if pred[2][0] > 0.5 else "NO")
