import numpy as np
import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

MAX_VOCAB_SIZE = 20000  
MAX_SEQUENCE_LENGTH = 300  

def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    asap_path = os.path.join(BASE_DIR, "data", "asap.csv")
    common_lit_path = os.path.join(BASE_DIR, "data", "common_lit.csv")
    jfleg_path = os.path.join(BASE_DIR, "data", "jfleg.csv")

    df_asp = pd.read_csv(asap_path)
    df_commonlit = pd.read_csv(common_lit_path)
    df_jfleg = pd.read_csv(jfleg_path)

    df = pd.DataFrame()
    df['text'] = pd.concat([df_asp['full_text'], df_commonlit['excerpt'], df_jfleg['sentence']], ignore_index=True)
    df['score'] = pd.concat([df_asp['score'], pd.Series([None] * (len(df_commonlit) + len(df_jfleg)))], ignore_index=True)
    df['score'] = df['score'].fillna(0)  # Wypełnianie brakujących wartości w score (lub inną wartością)

    df['readability_flesch'] = pd.concat([pd.Series([None] * len(df_asp)), df_commonlit['flesch_reading_ease'], pd.Series([None] * len(df_jfleg))], ignore_index=True)
    df['score'] = df['score'].fillna(0)  # Wypełnianie brakujących wartości w score (lub inną wartością)

    df['corrected_text'] = pd.concat([pd.Series([None] * (len(df_asp) + len(df_commonlit))), df_jfleg['corrections']], ignore_index=True)
    df['corrected_text'] = df['corrected_text'].fillna('')


    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'].dropna())
    sequences = tokenizer.texts_to_sequences(df['text'].dropna())

    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    scaler = MinMaxScaler()
    readability_features = df[['readability_flesch']].fillna(0)
    scaled_readability = scaler.fit_transform(readability_features)

    X_text = np.array(padded_sequences)
    X_readability = np.array(scaled_readability)
    y_score = df['score'].fillna(0).values

    min_length = min(len(X_text), len(X_readability), len(y_score))
    X_text = X_text[:min_length]
    X_readability = X_readability[:min_length]
    y_score = y_score[:min_length]

    # Sprawdzamy długości przed podziałem
    assert len(X_text) == len(X_readability) == len(y_score)

    X_train_text, X_test_text, X_train_read, X_test_read, y_train, y_test = train_test_split(
        X_text, X_readability, y_score, test_size=0.2, random_state=42
    )

    df_path = os.path.join(BASE_DIR, "data", "preprocessed_data.csv")
    df.to_csv(df_path, index=False)

    return (X_train_text, X_test_text, X_train_read, X_test_read, y_train, y_test, tokenizer)

if __name__ == "__main__":
    load_data()