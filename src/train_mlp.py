import numpy as np
import os
from model_mlp import build_mlp_multihead
from data_preparation import prepare_data
from evaluate import evaluate_model
from visualize import plot_visualizer, plot_mse, plot_all_metrics_comparison
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.models import load_model # type: ignore


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "mlp.keras")

data = prepare_data()

X_asap_train, X_asap_test, y_asap_train, y_asap_test = data["asap"]
X_commonlit_train, X_commonlit_test, y_commonlit_train, y_commonlit_test = data["commonlit"]
X_jfleg_train, X_jfleg_test, y_jfleg_train, y_jfleg_test = data["jfleg"]

def dummy(n, shape=(1,)):
    """Tworzy dummy dane dla nieużywanych outputów"""
    return np.zeros((n,) + shape)

# NEW MODEL OPTION
model = build_mlp_multihead()

## LOAD SAVED MODEL
##model = load_model(model_path)

# Early stopping callbacks - monitoruje odpowiednie loss dla każdego datasetu
early_stopping_asap = EarlyStopping(monitor='val_score_output_loss', patience=3, restore_best_weights=True, mode='min')
early_stopping_commonlit = EarlyStopping(monitor='val_readability_output_loss', patience=3, restore_best_weights=True, mode='min')
early_stopping_jfleg = EarlyStopping(monitor='val_jfleg_output_loss', patience=3, restore_best_weights=True, mode='min')

print("=== Trenowanie na danych ASAP (essay scoring) ===")
history_asap = model.fit(
    X_asap_train,  # Tylko tekst jako input
    {
        "score_output": y_asap_train,           # Prawdziwe scores
        "readability_output": dummy(len(X_asap_train)),  # Dummy data
        "jfleg_output": dummy(len(X_asap_train), shape=())  # Dummy data (shape=() dla binary)
    },
    epochs=10,
    batch_size=64,
    validation_data=(
        X_asap_test,  # Tylko tekst jako input
        {
            "score_output": y_asap_test,            # Prawdziwe scores
            "readability_output": dummy(len(X_asap_test)),   # Dummy data
            "jfleg_output": dummy(len(X_asap_test), shape=())   # Dummy data
        }
    ),
    callbacks=[early_stopping_asap],
    verbose=1
)

print("=== Trenowanie na danych CommonLit (readability) ===")
history_commonlit = model.fit(
    X_commonlit_train,  # Tylko tekst jako input
    {
        "score_output": dummy(len(X_commonlit_train), shape=()),     # Dummy data
        "readability_output": y_commonlit_train,    # Prawdziwe readability scores
        "jfleg_output": dummy(len(X_commonlit_train), shape=())     # Dummy data
    },
    epochs=10,
    batch_size=64,
    validation_data=(
        X_commonlit_test,  # Tylko tekst jako input
        {
            "score_output": dummy(len(X_commonlit_test), shape=()),      # Dummy data
            "readability_output": y_commonlit_test,     # Prawdziwe readability scores
            "jfleg_output": dummy(len(X_commonlit_test), shape=())      # Dummy data
        }
    ),
    callbacks=[early_stopping_commonlit],
    verbose=1
)

print("=== Trenowanie na danych JFLEG (grammar correction) ===")
history_jfleg = model.fit(
    X_jfleg_train,  # Tylko tekst jako input
    {
        "score_output": dummy(len(X_jfleg_train), shape=()),        # Dummy data
        "readability_output": dummy(len(X_jfleg_train)),            # Dummy data
        "jfleg_output": y_jfleg_train           # Prawdziwe binary labels
    },
    epochs=10,
    batch_size=64,
    validation_data=(
        X_jfleg_test,  # Tylko tekst jako input
        {
            "score_output": dummy(len(X_jfleg_test), shape=()),         # Dummy data
            "readability_output": dummy(len(X_jfleg_test)),             # Dummy data
            "jfleg_output": y_jfleg_test            # Prawdziwe binary labels
        }
    ),
    callbacks=[early_stopping_jfleg],
    verbose=1
)

print("=== Wizualizacja wyników trenowania ===")
plot_visualizer(history_asap, title="ASAP")
plot_visualizer(history_commonlit, title="CommonLit")
plot_visualizer(history_jfleg, title="JFLEG")

print("=== Ewaluacja modelu ===")
mse_asap, mse_commonlit = evaluate_model(model, data)

plot_mse([mse_asap, mse_commonlit], ["ASAP", "CommonLit"])

# Po zakończeniu wszystkich treningów
plot_all_metrics_comparison(
    [history_asap, history_commonlit, history_jfleg], 
    ["ASAP", "CommonLit", "JFLEG"]
)

print("=== Zapisywanie modelu ===")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)

print(f"Model zapisany w: {model_path}")