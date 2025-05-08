from tensorflow.keras import layers, Model, Input # type: ignore

VOCAB_SIZE = 20000
EMBEDDING_DIM = 128
READABILITY_INPUT_DIM = 1
MAX_SEQUENCE_LENGTH = 300

def build_cnn_multihead():
    text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name="text_input")
    x = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(text_input)
    x = layers.Conv1D(filters=64, kernel_size=5, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)

    read_input = Input(shape=(READABILITY_INPUT_DIM,), name="readability_input")
    concat = layers.concatenate([x, read_input])
    shared = layers.Dense(64, activation='relu')(concat)

    score_output = layers.Dense(1, name="score_output")(shared)
    readability_output = layers.Dense(1, name="readability_output")(shared)
    jfleg_output = layers.Dense(1, activation='sigmoid', name="jfleg_output")(x)  # jak wcześniej, tylko z CNN features

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
