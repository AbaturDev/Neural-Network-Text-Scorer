import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 300

def prepare_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = {
        "asap": os.path.join(BASE_DIR, "data", "asap.csv"),
        "commonlit": os.path.join(BASE_DIR, "data", "common_lit.csv"),
        "jfleg": os.path.join(BASE_DIR, "data", "jfleg.csv")
    }

    df_asap = pd.read_csv(paths["asap"])
    df_commonlit = pd.read_csv(paths["commonlit"])
    df_jfleg = pd.read_csv(paths["jfleg"])

    def extract_first_correction(corrections_str):
        try:
            corrections_list = eval(corrections_str)
            return corrections_list[0] if corrections_list else ""
        except:
            return ""

    jfleg_error_data = pd.DataFrame({
        "sentence": df_jfleg["sentence"],
        "has_errors": [1] * len(df_jfleg)
    })

    correct_sentences = df_jfleg["corrections"].apply(extract_first_correction)
    correct_sentences = correct_sentences[correct_sentences != ""]
    
    jfleg_correct_data = pd.DataFrame({
        "sentence": correct_sentences,
        "has_errors": [0] * len(correct_sentences)
    })

    df_jfleg_balanced = pd.concat([jfleg_error_data, jfleg_correct_data], ignore_index=True)

    all_texts = pd.concat([
        df_asap["full_text"],
        df_commonlit["excerpt"],
        df_jfleg_balanced["sentence"]
    ]).dropna().astype(str)

    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(all_texts)

    def process_texts(texts):
        sequences = tokenizer.texts_to_sequences(texts.astype(str))
        return pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    X_asap = process_texts(df_asap["full_text"])
    df_asap_clean = pd.DataFrame({
        "text": df_asap["full_text"].astype(str),
        "X": list(X_asap),
        "y_asap": df_asap["score"],
        "y_commonlit": np.nan,
        "y_jfleg": np.nan
    })

    X_commonlit = process_texts(df_commonlit["excerpt"])
    scaler = MinMaxScaler()
    flesch_scaled = scaler.fit_transform(df_commonlit[["flesch_reading_ease"]].fillna(0)).flatten()
    df_commonlit_clean = pd.DataFrame({
        "text": df_commonlit["excerpt"].astype(str),
        "X": list(X_commonlit),
        "y_asap": np.nan,
        "y_commonlit": flesch_scaled,
        "y_jfleg": np.nan
    })

    X_jfleg = process_texts(df_jfleg_balanced["sentence"])
    df_jfleg_clean = pd.DataFrame({
        "text": df_jfleg_balanced["sentence"].astype(str),
        "X": list(X_jfleg),
        "y_asap": np.nan,
        "y_commonlit": np.nan,
        "y_jfleg": df_jfleg_balanced["has_errors"]
    })

    df_all = pd.concat([df_asap_clean, df_commonlit_clean, df_jfleg_clean], ignore_index=True)
    
    df_train, df_temp = train_test_split(df_all, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

    def extract_data(df):
        X = np.stack(df["X"].values)
        
        def get_target_and_mask(col):
            y = df[col].fillna(0.0).values.astype(np.float32)
            mask = (~df[col].isna()).astype(np.float32).values
            return y, mask

        y_asap, mask_asap = get_target_and_mask("y_asap")
        y_commonlit, mask_commonlit = get_target_and_mask("y_commonlit")
        y_jfleg, mask_jfleg = get_target_and_mask("y_jfleg")

        return X, {
            "score_output": y_asap,
            "readability_output": y_commonlit,
            "jfleg_output": y_jfleg
        }, {
            "score_output": mask_asap,
            "readability_output": mask_commonlit,
            "jfleg_output": mask_jfleg
        }

    X_train, y_train, sw_train = extract_data(df_train)
    X_val, y_val, sw_val = extract_data(df_val)
    X_test, y_test, sw_test = extract_data(df_test)

    return {
        "tokenizer": tokenizer,
        "scaler": scaler,
        "train": (X_train, y_train, sw_train),
        "validation": (X_val, y_val, sw_val),
        "test": (X_test, y_test, sw_test)
    }