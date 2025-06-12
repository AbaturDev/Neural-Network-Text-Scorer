import numpy as np
import os
from model_mlp import build_mlp_multihead
from data_preparation import prepare_data
from evaluate import evaluate_model
from visualize import plot_visualizer, plot_mse, plot_all_metrics_comparison
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.models import load_model # type: ignore


EPOCHS = 20
BATCH_SIZE = 64

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "mlp.keras")

data = prepare_data()

X_train, y_train, sw_train = data["train"]
X_val, y_val, sw_val = data["validation"]
X_test, y_test, sw_test = data["test"]

def dummy(n, shape=(1,)):
    return np.zeros((n,) + shape, dtype=np.float32)

# NEW MODEL OPTION
model = build_mlp_multihead()

# LOAD SAVED MODEL
#model = load_model(model_path)

early_stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, min_delta=0.001)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-6,
    verbose=1
)

sw_train_list = [sw_train["score_output"], sw_train["readability_output"], sw_train["jfleg_output"]]
sw_val_list = [sw_val["score_output"], sw_val["readability_output"], sw_val["jfleg_output"]]  # NOWY

y_train_list = [y_train["score_output"], y_train["readability_output"], y_train["jfleg_output"]]
y_val_list = [y_val["score_output"], y_val["readability_output"], y_val["jfleg_output"]]      # NOWY

history = model.fit(
    X_train,
    y_train_list,
    sample_weight=sw_train_list,
    validation_data=(X_val, y_val_list, sw_val_list),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

evaluate_model(model, data)

os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)