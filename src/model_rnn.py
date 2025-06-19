import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.regularizers import l2

VOCAB_SIZE = 20000
EMBEDDING_DIM = 256

def build_rnn_multihead():
    text_input = Input(shape=(300,), name="text_input")

    embedded = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(text_input)
    embedded = layers.SpatialDropout1D(0.3)(embedded)

    lstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(embedded)
    lstm = layers.GlobalMaxPooling1D()(lstm)

    shared = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001))(lstm)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.4)(shared)

    shared = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001))(shared)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.3)(shared)

    asap = layers.Dense(32, activation='relu')(shared)
    asap = layers.Dropout(0.2)(asap)
    score_output = layers.Dense(1, name="score_output")(asap)

    commonlit = layers.Dense(32, activation='relu')(shared)
    commonlit = layers.Dropout(0.2)(commonlit)
    readability_output = layers.Dense(1, name="readability_output")(commonlit)

    jfleg = layers.Dense(64, activation='relu')(shared)
    jfleg = layers.Dropout(0.3)(jfleg)
    jfleg = layers.Dense(32, activation='relu')(jfleg)
    jfleg_output = layers.Dense(1, activation='sigmoid', name="jfleg_output")(jfleg)

    model = Model(inputs=text_input, outputs=[score_output, readability_output, jfleg_output])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    
    loss_weights={
        "score_output": 5.0,
        "readability_output": 1.0,
        "jfleg_output": 8.0
    }

    model.compile(
        optimizer=optimizer,
        loss={
            "score_output": "mse",
            "readability_output": "mse",
            "jfleg_output": "binary_crossentropy"
        },
        loss_weights=loss_weights,
        metrics={
            "score_output": ["mae"],
            "readability_output": ["mae"],
            "jfleg_output": ["accuracy", "precision", "recall"]
        }
    )

    return model, loss_weights
