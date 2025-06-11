import tensorflow as tf
from tensorflow.keras import layers, Model, Input # type: ignore

VOCAB_SIZE = 20000
EMBEDDING_DIM = 128

def build_mlp_multihead():
    # Jeden input - tekst
    text_input = Input(shape=(300,), name="text_input")
    
    # Wspólne przetwarzanie tekstu
    embedded = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(text_input)
    pooled = layers.GlobalAveragePooling1D()(embedded)
    
    # Wspólne features z tekstu
    shared_features = layers.Dense(128, activation='relu', name='shared_features')(pooled)
    shared_features = layers.Dropout(0.3)(shared_features)
    
    # Dodatkowa warstwa wspólna
    shared_features_2 = layers.Dense(64, activation='relu', name='shared_features_2')(shared_features)
    shared_features_2 = layers.Dropout(0.3)(shared_features_2)
    
    # HEAD 1: ASAP - Ocena esejów (regresja)
    asap_dense = layers.Dense(32, activation='relu', name='asap_head')(shared_features_2)
    score_output = layers.Dense(1, name="score_output")(asap_dense)
    
    # HEAD 2: CommonLit - Czytelność (regresja) 
    commonlit_dense = layers.Dense(32, activation='relu', name='commonlit_head')(shared_features_2)
    readability_output = layers.Dense(1, name="readability_output")(commonlit_dense)
    
    # HEAD 3: JFLEG - Błędy gramatyczne (klasyfikacja binarna)
    jfleg_dense = layers.Dense(32, activation='relu', name='jfleg_head')(shared_features_2)
    jfleg_output = layers.Dense(1, activation='sigmoid', name="jfleg_output")(jfleg_dense)
    
    # Model z jednym inputem i trzema outputami
    model = Model(
        inputs=text_input,  # TYLKO tekst
        outputs=[score_output, readability_output, jfleg_output]
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "score_output": "mse",                    # ASAP - regresja
            "readability_output": "mse",              # CommonLit - regresja  
            "jfleg_output": "binary_crossentropy"     # JFLEG - klasyfikacja
        },
        loss_weights={
            "score_output": 1.0,
            "readability_output": 1.0,
            "jfleg_output": 1.0
        },
        metrics={
            "score_output": ["mae"],
            "readability_output": ["mae"],
            "jfleg_output": ["accuracy"]
        }
    )
    
    return model
