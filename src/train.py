import numpy as np
import os
from model_mlp import build_mlp_multihead
from data_preparation import prepare_data
from evaluate import evaluate_model
from visualize import plot_visualizer

data = prepare_data()

X_asap_train, X_asap_test, y_asap_train, y_asap_test = data["asap"]
X_commonlit_train, X_commonlit_test, y_commonlit_train, y_commonlit_test = data["commonlit"]
X_jfleg_train, X_jfleg_test, y_jfleg_train, y_jfleg_test = data["jfleg"]

y_jfleg_train = np.array([1 if str(c).strip() != "" else 0 for c in y_jfleg_train])
y_jfleg_test = np.array([1 if str(c).strip() != "" else 0 for c in y_jfleg_test])

def dummy(n, shape=(1,)):
    return np.zeros((n,) + shape)

model = build_mlp_multihead()

history_asap = model.fit(
    {"text_input": X_asap_train, "readability_input": dummy(len(X_asap_train))},
    {
        "score_output": y_asap_train,
        "readability_output": dummy(len(X_asap_train)),
        "jfleg_output": dummy(len(X_asap_train), shape=())
    },
    epochs=5,
    batch_size=32,
    validation_data=(
        {"text_input": X_asap_test, "readability_input": dummy(len(X_asap_test))},
        {
            "score_output": y_asap_test,
            "readability_output": dummy(len(X_asap_test)),
            "jfleg_output": dummy(len(X_asap_test), shape=())
        }
    )
)

history_commonlit = model.fit(
    {"text_input": X_commonlit_train, "readability_input": y_commonlit_train},
    {
        "score_output": dummy(len(X_commonlit_train), shape=()),
        "readability_output": y_commonlit_train,
        "jfleg_output": dummy(len(X_commonlit_train), shape=())
    },
    epochs=5,
    batch_size=32,
    validation_data=(
        {"text_input": X_commonlit_test, "readability_input": y_commonlit_test},
        {
            "score_output": dummy(len(X_commonlit_test), shape=()),
            "readability_output": y_commonlit_test,
            "jfleg_output": dummy(len(X_commonlit_test), shape=())
        }
    )
)

history_jfleg = model.fit(
    {"text_input": X_jfleg_train, "readability_input": dummy(len(X_jfleg_train))},
    {
        "score_output": dummy(len(X_jfleg_train), shape=()),
        "readability_output": dummy(len(X_jfleg_train)),
        "jfleg_output": y_jfleg_train
    },
    epochs=5,
    batch_size=32,
    validation_data=(
        {"text_input": X_jfleg_test, "readability_input": dummy(len(X_jfleg_test))},
        {
            "score_output": dummy(len(X_jfleg_test), shape=()),
            "readability_output": dummy(len(X_jfleg_test)),
            "jfleg_output": y_jfleg_test
        }
    )
)

plot_visualizer(history_asap, title="ASAP")
plot_visualizer(history_commonlit, title="CommonLit")
plot_visualizer(history_jfleg, title="JFLEG")

model_path = os.path.join("models", "mlp.keras")
os.makedirs(os.path.dirname(model_path), exist_ok=True)

model.save(model_path)
