from sklearn.metrics import mean_squared_error, classification_report
import numpy as np

def evaluate_model(model, data):
    _, X_test_asap, _, y_test_asap = data["asap"]
    _, X_test_commonlit, _, y_test_commonlit = data["commonlit"]
    _, X_test_jfleg, _, y_test_jfleg = data["jfleg"]

    preds_asap = model.predict(X_test_asap, verbose=0)
    y_pred_asap = preds_asap[0].flatten()  # score_output (pierwszy output)
    mse_asap = mean_squared_error(y_test_asap, y_pred_asap)
    preds_commonlit = model.predict(X_test_commonlit, verbose=0)
    y_pred_commonlit = preds_commonlit[1].flatten()  # readability_output (drugi output)
    mse_commonlit = mean_squared_error(y_test_commonlit, y_pred_commonlit)
    preds_jfleg = model.predict(X_test_jfleg, verbose=0)
    y_pred_jfleg = preds_jfleg[2].flatten()  # jfleg_output (trzeci output)
    
    y_pred_classes = (y_pred_jfleg > 0.5).astype(int)
    y_true_classes = y_test_jfleg.astype(int)

    accuracy = np.mean(y_pred_classes == y_true_classes)
    print(f"[JFLEG] Accuracy: {accuracy:.4f}")

    return mse_asap, mse_commonlit