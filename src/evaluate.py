from sklearn.metrics import mean_squared_error, classification_report
import numpy as np

def evaluate_model(model, data):
    """
    Ewaluuje model na wszystkich trzech taskach.
    Model ma teraz tylko jeden input (tekst), więc nie przekazujemy słownika.
    """
    _, X_test_asap, _, y_test_asap = data["asap"]
    _, X_test_commonlit, _, y_test_commonlit = data["commonlit"]
    _, X_test_jfleg, _, y_test_jfleg = data["jfleg"]

    print("=== Ewaluacja ASAP (Essay Scoring) ===")
    # Predykcja dla ASAP - tylko tekst jako input
    preds_asap = model.predict(X_test_asap, verbose=0)
    # preds_asap to lista [score_output, readability_output, jfleg_output]
    y_pred_asap = preds_asap[0].flatten()  # score_output (pierwszy output)
    mse_asap = mean_squared_error(y_test_asap, y_pred_asap)
    print(f"[ASAP] Mean Squared Error (MSE): {mse_asap:.4f}")

    print("=== Ewaluacja CommonLit (Readability) ===")
    # Predykcja dla CommonLit - tylko tekst jako input
    preds_commonlit = model.predict(X_test_commonlit, verbose=0)
    y_pred_commonlit = preds_commonlit[1].flatten()  # readability_output (drugi output)
    mse_commonlit = mean_squared_error(y_test_commonlit, y_pred_commonlit)
    print(f"[CommonLit] Mean Squared Error (MSE): {mse_commonlit:.4f}")

    print("=== Ewaluacja JFLEG (Grammar Correction) ===")
    # Predykcja dla JFLEG - tylko tekst jako input
    preds_jfleg = model.predict(X_test_jfleg, verbose=0)
    y_pred_jfleg = preds_jfleg[2].flatten()  # jfleg_output (trzeci output)
    
    # Konwersja na klasy binarne (próg 0.5)
    y_pred_classes = (y_pred_jfleg > 0.5).astype(int)
    y_true_classes = y_test_jfleg.astype(int)

    print("[JFLEG] Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, digits=4))
    
    # Dodatkowe metryki dla JFLEG
    accuracy = np.mean(y_pred_classes == y_true_classes)
    print(f"[JFLEG] Accuracy: {accuracy:.4f}")

    return mse_asap, mse_commonlit