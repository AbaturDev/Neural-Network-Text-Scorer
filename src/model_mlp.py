import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.regularizers import l2

VOCAB_SIZE = 20000
EMBEDDING_DIM = 256

def build_mlp_multihead():
    text_input = Input(shape=(300,), name="text_input")
    
    embedded = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(text_input)
    
    embedded = layers.Dropout(0.2)(embedded)
    
    avg_pool = layers.GlobalAveragePooling1D()(embedded)
    max_pool = layers.GlobalMaxPooling1D()(embedded)
    
    pooled = layers.Concatenate()([avg_pool, max_pool])
    
    shared = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001))(pooled)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.4)(shared)
    
    shared = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001))(shared)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.3)(shared)
    
    shared = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001))(shared)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.3)(shared)
    
    asap_head = layers.Dense(64, activation='relu', name='asap_head_1')(shared)
    asap_head = layers.BatchNormalization()(asap_head)
    asap_head = layers.Dropout(0.2)(asap_head)
    asap_head = layers.Dense(32, activation='relu', name='asap_head_2')(asap_head)
    score_output = layers.Dense(1, name="score_output")(asap_head)
    
    commonlit_head = layers.Dense(64, activation='relu', name='commonlit_head_1')(shared)
    commonlit_head = layers.BatchNormalization()(commonlit_head)
    commonlit_head = layers.Dropout(0.2)(commonlit_head)
    commonlit_head = layers.Dense(32, activation='relu', name='commonlit_head_2')(commonlit_head)
    readability_output = layers.Dense(1, name="readability_output")(commonlit_head)
    
    jfleg_head = layers.Dense(128, activation='relu')(shared)
    jfleg_head = layers.BatchNormalization()(jfleg_head)
    jfleg_head = layers.Dropout(0.4)(jfleg_head)
    jfleg_head = layers.Dense(64, activation='relu')(jfleg_head)
    jfleg_head = layers.BatchNormalization()(jfleg_head)
    jfleg_head = layers.Dropout(0.3)(jfleg_head)
    jfleg_head = layers.Dense(32, activation='relu')(jfleg_head)
    jfleg_head = layers.BatchNormalization()(jfleg_head)
    jfleg_head = layers.Dropout(0.2)(jfleg_head)
    jfleg_output = layers.Dense(1, activation='sigmoid', name="jfleg_output")(jfleg_head)
    
    model = Model(
        inputs=text_input,
        outputs=[score_output, readability_output, jfleg_output]
    )
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0
    )
    
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