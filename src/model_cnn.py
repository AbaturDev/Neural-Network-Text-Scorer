import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.regularizers import l2

VOCAB_SIZE = 20000
EMBEDDING_DIM = 256

def build_cnn_multihead():
    text_input = Input(shape=(300,), name="text_input")

    embedded = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(text_input)
    embedded = layers.SpatialDropout1D(0.2)(embedded)

    conv1 = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(embedded)
    conv2 = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(embedded)
    conv3 = layers.Conv1D(128, kernel_size=7, activation='relu', padding='same')(embedded)

    avg_pool = layers.GlobalAveragePooling1D()(conv1)
    max_pool = layers.GlobalMaxPooling1D()(conv2)
    sum_pool = layers.GlobalMaxPooling1D()(conv3)

    pooled = layers.Concatenate()([avg_pool, max_pool, sum_pool])

    shared = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001))(pooled)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.4)(shared)

    shared = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001))(shared)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.3)(shared)

    shared = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001))(shared)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.3)(shared)

    # Głowy
    asap = layers.Dense(64, activation='relu')(shared)
    asap = layers.BatchNormalization()(asap)
    asap = layers.Dropout(0.2)(asap)
    asap = layers.Dense(32, activation='relu')(asap)
    score_output = layers.Dense(1, name="score_output")(asap)

    commonlit = layers.Dense(64, activation='relu')(shared)
    commonlit = layers.BatchNormalization()(commonlit)
    commonlit = layers.Dropout(0.2)(commonlit)
    commonlit = layers.Dense(32, activation='relu')(commonlit)
    readability_output = layers.Dense(1, name="readability_output")(commonlit)

    jfleg = layers.Dense(128, activation='relu')(shared)
    jfleg = layers.BatchNormalization()(jfleg)
    jfleg = layers.Dropout(0.4)(jfleg)
    jfleg = layers.Dense(64, activation='relu')(jfleg)
    jfleg = layers.BatchNormalization()(jfleg)
    jfleg = layers.Dropout(0.3)(jfleg)
    jfleg = layers.Dense(32, activation='relu')(jfleg)
    jfleg = layers.BatchNormalization()(jfleg)
    jfleg = layers.Dropout(0.2)(jfleg)
    jfleg_output = layers.Dense(1, activation='sigmoid', name="jfleg_output")(jfleg)

    model = Model(inputs=text_input, outputs=[score_output, readability_output, jfleg_output])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss={
            "score_output": "mse",
            "readability_output": "mse",
            "jfleg_output": "binary_crossentropy"
        },
        loss_weights={
            "score_output": 15.0,
            "readability_output": 1.0,
            "jfleg_output": 8.0
        },
        metrics={
            "score_output": ["mae"],
            "readability_output": ["mae"],
            "jfleg_output": ["accuracy", "precision", "recall"]
        }
    )

    return model
