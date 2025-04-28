from sklearn.metrics import mean_squared_error, classification_report
import numpy as np

def evaluate_model(model, data):
    _, X_test_asap, _, y_test_asap = data["asap"]
    _, X_test_commonlit, _, y_test_commonlit = data["commonlit"]
    _, X_test_jfleg, _, y_test_jfleg = data["jfleg"]

    y_test_jfleg = np.array([1 if str(c).strip() != "" else 0 for c in y_test_jfleg])

    preds_asap = model.predict(
        {"text_input": X_test_asap, "readability_input": np.zeros((len(X_test_asap), 1))},
        verbose=0
    )
    y_pred_asap = preds_asap[0].flatten()
    mse_asap = mean_squared_error(y_test_asap, y_pred_asap)
    print(f"[ASAP] Mean Squared Error (MSE): {mse_asap:.4f}")

    preds_commonlit = model.predict(
        {"text_input": X_test_commonlit, "readability_input": y_test_commonlit.reshape(-1, 1)},
        verbose=0
    )
    y_pred_commonlit = preds_commonlit[1].flatten()
    mse_commonlit = mean_squared_error(y_test_commonlit, y_pred_commonlit)
    print(f"[CommonLit] Mean Squared Error (MSE): {mse_commonlit:.4f}")

    preds_jfleg = model.predict(
        {"text_input": X_test_jfleg, "readability_input": np.zeros((len(X_test_jfleg), 1))},
        verbose=0
    )
    y_pred_jfleg = preds_jfleg[2].flatten()
    y_pred_classes = (y_pred_jfleg > 0.5).astype(int)
    y_true_classes = y_test_jfleg.astype(int)

    print("[JFLEG] Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, digits=4))

    return mse_asap, mse_commonlit
