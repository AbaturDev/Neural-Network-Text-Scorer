import numpy as np
import os
from model_rnn import build_rnn_multihead
from data_preparation import prepare_data
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import load_model
from evaluate import evaluate_model


def dummy(n, shape=(1,)):
    return np.zeros((n,) + shape, dtype=np.float32)


EPOCHS = 20
BATCH_SIZE = 64

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "rnn.keras")

data = prepare_data()

X_train, y_train, sw_train = data["train"]
X_val, y_val, sw_val = data["validation"]
X_test, y_test, sw_test = data["test"]


# NEW MODEL OPTION
model = build_rnn_multihead()

# LOAD SAVED MODEL OPTION
# model = load_model(model_path)

early_stop = EarlyStopping(
    monitor="val_jfleg_output_accuracy",
    patience=10,
    restore_best_weights=True,
    min_delta=0.005,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=4,
    min_lr=1e-7,
    verbose=1,
    cooldown=2
)

checkpoint = ModelCheckpoint(
    model_path,
    monitor='val_jfleg_output_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr * 0.9
    else:
        return lr * 0.8

lr_scheduler = LearningRateScheduler(scheduler, verbose=1)

sw_train_list = [sw_train["score_output"], sw_train["readability_output"], sw_train["jfleg_output"]]
sw_val_list = [sw_val["score_output"], sw_val["readability_output"], sw_val["jfleg_output"]]

y_train_list = [y_train["score_output"], y_train["readability_output"], y_train["jfleg_output"]]
y_val_list = [y_val["score_output"], y_val["readability_output"], y_val["jfleg_output"]]

history = model.fit(
    X_train,
    y_train_list,
    sample_weight=sw_train_list,
    validation_data=(X_val, y_val_list, sw_val_list),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, reduce_lr, checkpoint, lr_scheduler],
    verbose=1,
)

evaluate_model(model, data)

os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)
