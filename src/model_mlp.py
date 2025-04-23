import tensorflow as tf
from tensorflow.keras import layers, Model, Input # type: ignore

VOCAB_SIZE = 20000
EMBEDDING_DIM = 128
READABILITY_INPUT_DIM = 1

def build_mlp_multihead():
    text_input = Input(shape=(300,), name="text_input")
    x = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(text_input)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)

    read_input = Input(shape=(READABILITY_INPUT_DIM,), name="readability_input")
    concat = layers.concatenate([x, read_input])

    shared = layers.Dense(64, activation='relu')(concat)

    score_output = layers.Dense(1, name="score_output")(shared)

    readability_output = layers.Dense(1, name="readability_output")(shared)

    jfleg_output = layers.Dense(1, activation='sigmoid', name="jfleg_output")(x)

    model = Model(inputs=[text_input, read_input], outputs=[score_output, readability_output, jfleg_output])
    model.compile(
        optimizer='adam',
        loss={
            "score_output": "mse",
            "readability_output": "mse",
            "jfleg_output": "binary_crossentropy"
        },
        loss_weights={
            "score_output": 1.0,
            "readability_output": 1.0,
            "jfleg_output": 1.0
        },
        metrics={"score_output": "mae", "readability_output": "mae", "jfleg_output": "accuracy"}
    )
    return model
