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

    all_texts = pd.concat([
        df_asap["full_text"],
        df_commonlit["excerpt"],
        df_jfleg["sentence"]
    ]).dropna().astype(str)

    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(all_texts)

    def process_texts(texts):
        sequences = tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    # ASAP
    X_asap = process_texts(df_asap["full_text"].astype(str))
    y_asap = df_asap["score"].values

    # CommonLit
    X_commonlit = process_texts(df_commonlit["excerpt"].astype(str))
    scaler = MinMaxScaler()
    y_commonlit = scaler.fit_transform(df_commonlit[["flesch_reading_ease"]].fillna(0))

    # JFLEG
    X_jfleg = process_texts(df_jfleg["sentence"].astype(str))
    y_jfleg = df_jfleg["corrections"].astype(str)

    return {
        "tokenizer": tokenizer,
        "asap": train_test_split(X_asap, y_asap, test_size=0.2, random_state=42),
        "commonlit": train_test_split(X_commonlit, y_commonlit, test_size=0.2, random_state=42),
        "jfleg": train_test_split(X_jfleg, y_jfleg, test_size=0.2, random_state=42)
    }
