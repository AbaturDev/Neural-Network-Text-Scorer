from sklearn.metrics import mean_squared_error, classification_report
import numpy as np

def evaluate_model(model, data):
    _, X_test_asap, _, y_test_asap = data["asap"]
    _, X_test_commonlit, _, y_test_commonlit = data["commonlit"]
    _, X_test_jfleg, _, y_test_jfleg = data["jfleg"]

    predictions = model.predict({
        "asap_input": X_test_asap,
        "commonlit_input": X_test_commonlit,
        "jfleg_input": X_test_jfleg
    })

    y_pred_asap = predictions["score_output"].flatten()
    mse_asap = mean_squared_error(y_test_asap, y_pred_asap)
    print(f"[ASAP] Mean Squared Error: {mse_asap:.4f}")

    y_pred_commonlit = predictions["readability_output"].flatten()
    mse_commonlit = mean_squared_error(y_test_commonlit, y_pred_commonlit)
    print(f"[CommonLit] Mean Squared Error: {mse_commonlit:.4f}")

    y_pred_jfleg = predictions["jfleg_output"]
    y_pred_classes = np.argmax(y_pred_jfleg, axis=1)
    y_true_classes = y_test_jfleg.astype(int)

    print("[JFLEG] Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes))
