import streamlit as st
import numpy as np
import fitz
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_preparation import prepare_data

st.set_page_config(
    page_title="Text Scorer",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_LEN = 300

MODELS = {
    "MLP (Multi-Layer Perceptron)": "mlp.keras",
    "CNN (Convolutional Neural Network)": "cnn.keras", 
    "RNN (Recurrent Neural Network)": "rnn.keras"
}

@st.cache_resource
def load_selected_model(model_name):
    model_path = os.path.join(BASE_DIR, "models", model_name)
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model {model_name}: {str(e)}")
        return None

@st.cache_data
def load_tokenizer():
    data = prepare_data()
    return data["tokenizer"]

def preprocess_text(text, tokenizer):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded

def extract_text(file):
    try:
        if file.name.endswith(".txt"):
            return file.read().decode("utf-8")
        elif file.name.endswith(".pdf"):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            return "\n".join([page.get_text() for page in doc])
        else:
            return ""
    except Exception as e:
        st.error(f"Failed to load the file: {str(e)}")
        return ""

def display_score_explanations():
    st.markdown("### 📚 Scoring Scale Explanations:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📘 ASAP (Automated Student Assessment Prize)**
        - **Range:** 0-6 points
        - **Description:** Overall essay quality assessment
        - **Interpretation:**
          - 0-2: Bad
          - 2-3.5: Average
          - 3.5-5: Good
          - 5-6: Excellent
        """)
    
    with col2:
        st.markdown("""
        **📗 CommonLit Readability**
        - **Range:** 0-1
        - **Description:** Text readability level
        - **Interpretation:**
          - 0-0.3: Difficult
          - 0.3-0.7: Average
          - 0.7-1.0: Easy
        """)
    
    with col3:
        st.markdown("""
        **📕 JFLEG (JHU FLuency-Extended GUG)**
        - **Range:** 0-1
        - **Description:** Grammatical correctness
        - **Interpretation:**
          - 0.0-0.3: Many errors
          - 0.4-0.7: Moderately correct
          - 0.8-1.0: Very correct
        """)

def display_results(predictions):
    st.markdown("### 📊 Analysis Results:")
    
    col1, col2, col3 = st.columns(3)
    
    asap_score = float(predictions[0][0])
    commonlit_score = float(predictions[1][0])
    jfleg_score = float(predictions[2][0])
    
    with col1:
        if asap_score <= 2:
            color = "🔴"
            level = "Poor"
        elif asap_score <= 3.5:
            color = "🟡"
            level = "Average"
        elif asap_score <= 5:
            color = "🟠"
            level = "Good"
        else:
            color = "🟢"
            level = "Excellent"
        
        st.metric(
            label="📘 ASAP (Essay Quality)",
            value=f"{asap_score:.2f}/6",
            help="Overall quality and structure of the text"
        )
        st.markdown(f"{color} **Level:** {level}")
    
    with col2:
        if commonlit_score <= 0.3:
            color = "🔴"
            level = "Very difficult"
        elif commonlit_score <= 0.7:
            color = "🟡"
            level = "Average"
        else:
            color = "🟢"
            level = "Easy"
        
        st.metric(
            label="📗 CommonLit (Readability)",
            value=f"{commonlit_score:.3f}",
            help="Text difficulty level for readers"
        )
        st.markdown(f"{color} **Level:** {level}")
    
    with col3:
        if jfleg_score <= 0.3:
            color = "🔴"
            level = "Many errors"
        elif jfleg_score <= 0.7:
            color = "🟡"
            level = "Moderately correct"
        else:
            color = "🟢"
            level = "Very correct"
        
        st.metric(
            label="📕 JFLEG (Grammar)",
            value=f"{jfleg_score:.3f}",
            help="Grammatical correctness and text fluency"
        )
        st.markdown(f"{color} **Level:** {level}")

st.title("📊 TEXT SCORER")
st.markdown("Analyze text quality using different machine learning models")

st.sidebar.header("⚙️ Configuration")
selected_model_name = st.sidebar.selectbox(
    "Select analysis model:",
    list(MODELS.keys()),
    help="Each model has a different approach to text analysis"
)

model_info = {
    "MLP (Multi-Layer Perceptron)": "Classic neural network with hidden layers. Good for basic text analysis.",
    "CNN (Convolutional Neural Network)": "Convolutional network, effective at recognizing local patterns in text.",
    "RNN (Recurrent Neural Network)": "Recurrent network, considers sequence and context in text."
}

st.sidebar.info(f"**Model description:** {model_info[selected_model_name]}")

model = load_selected_model(MODELS[selected_model_name])
tokenizer = load_tokenizer()

if model is None:
    st.error("Failed to load model. Check if the file exists.")
    st.stop()

st.markdown("### 📁 Upload file for analysis")
uploaded_file = st.file_uploader(
    "Choose a file (.txt or .pdf)",
    type=["txt", "pdf"],
    help="Supported formats: text files (.txt) and PDF documents (.pdf)"
)

if uploaded_file is not None:
    st.info(f"📄 File loaded: **{uploaded_file.name}** ({uploaded_file.size} bytes)")
    
    with st.spinner("Reading file content..."):
        text = extract_text(uploaded_file)
    
    if text.strip() == "":
        st.error("❌ Failed to read file content or file is empty.")
    else:
        st.success("✅ Text successfully read!")
        
        with st.expander("👀 Text preview (first 500 characters)"):
            preview_text = text[:500] + "..." if len(text) > 500 else text
            st.text(preview_text)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Characters", len(text))
        with col2:
            st.metric("Words", len(text.split()))
        with col3:
            st.metric("Sentences", text.count('.') + text.count('!') + text.count('?'))
        with col4:
            st.metric("Paragraphs", len([p for p in text.split('\n\n') if p.strip()]))
        
        if st.button("🔍 Analyze text", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing text using {selected_model_name} model..."):
                try:
                    X = preprocess_text(text, tokenizer)
                    
                    predictions = model.predict(X)
                    
                    display_results(predictions)
                    
                    st.markdown("---")
                    asap_score = float(predictions[0][0])
                    commonlit_score = float(predictions[1][0])
                    jfleg_score = float(predictions[2][0])
                    
                    st.markdown("### 📋 Summary:")
                    
                    if asap_score <= 2:
                        overall_quality = "poor"
                    elif asap_score <= 3.5:
                        overall_quality = "average"
                    elif asap_score <= 5:
                        overall_quality = "good"
                    else:
                        overall_quality = "excellent"
                    
                    if commonlit_score <= 0.3:
                        readability = "very difficult"
                    elif commonlit_score <= 0.7:
                        readability = "average difficulty"
                    else:
                        readability = "easy"
                    
                    if jfleg_score <= 0.3:
                        grammar_quality = "poor"
                    elif jfleg_score <= 0.7:
                        grammar_quality = "moderate"
                    else:
                        grammar_quality = "very good"
                    
                    st.markdown(f"""
                    **The analyzed text is characterized by:**
                    - **Overall quality:** {overall_quality} ({asap_score:.2f}/6 pts.)
                    - **Readability:** {readability} ({commonlit_score:.3f} pts.)
                    - **Grammatical correctness:** {grammar_quality} ({jfleg_score:.3f} pts.)
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Analysis error: {str(e)}")

st.markdown("---")
display_score_explanations()

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    📊 Text Scorer - Automatic text quality assessment tool<br>
    Select a model, upload a file and get detailed analysis!
    </div>
    """, 
    unsafe_allow_html=True
)