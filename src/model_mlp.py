import tensorflow as tf
from tensorflow.keras import layers, Model, Input # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore

VOCAB_SIZE = 20000
EMBEDDING_DIM = 256

def build_mlp_multihead():
    text_input = Input(shape=(300,), name="text_input")
    
    embedded = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)(text_input)
    pooled = layers.GlobalAveragePooling1D()(embedded)
    
    shared_features = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001), name='shared_features')(pooled)
    shared_features = layers.BatchNormalization()(shared_features)
    shared_features = layers.Dropout(0.3)(shared_features)
    
    shared_features_2 = layers.Dense(64, activation='relu', name='shared_features_2')(shared_features)
    shared_features_2 = layers.Dropout(0.3)(shared_features_2)
    
    asap_dense = layers.Dense(32, activation='relu', name='asap_head')(shared_features_2)
    score_output = layers.Dense(1, name="score_output")(asap_dense)
    
    commonlit_dense = layers.Dense(32, activation='relu', name='commonlit_head')(shared_features_2)
    readability_output = layers.Dense(1, name="readability_output")(commonlit_dense)
    
    jfleg_dense = layers.Dense(32, activation='relu', name='jfleg_head')(shared_features_2)
    jfleg_output = layers.Dense(1, activation='sigmoid', name="jfleg_output")(jfleg_dense)
    
    model = Model(
        inputs=text_input,
        outputs=[score_output, readability_output, jfleg_output]
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss={
            "score_output": "mse",
            "readability_output": "mse",  
            "jfleg_output": "binary_crossentropy"
        },
        loss_weights={
            "score_output": 0.1,
            "readability_output": 1.0,
            "jfleg_output": 5.0
        },
        metrics={
            "score_output": ["mae"],
            "readability_output": ["mae"],
            "jfleg_output": ["accuracy"]
        }
    )
    
    return model
