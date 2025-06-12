from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
import numpy as np
import pandas as pd

def evaluate_model(model, data):
    X_test, y_test, sw_test = data["test"]
    
    predictions = model.predict(X_test, verbose=0)
    
    results = {}
    
    asap_mask = sw_test["score_output"] == 1
    if asap_mask.sum() > 0:
        y_true_asap = y_test["score_output"][asap_mask]
        y_pred_asap = predictions[0][asap_mask].flatten()
        
        mse_asap = mean_squared_error(y_true_asap, y_pred_asap)
        mae_asap = np.mean(np.abs(y_true_asap - y_pred_asap))
        
        results["asap"] = {
            "mse": mse_asap,
            "mae": mae_asap,
            "rmse": np.sqrt(mse_asap),
            "n_samples": asap_mask.sum()
        }
        
        print(f"[ASAP] Samples: {asap_mask.sum()}")
        print(f"[ASAP] MSE: {mse_asap:.4f}")
        print(f"[ASAP] MAE: {mae_asap:.4f}")
        print(f"[ASAP] RMSE: {np.sqrt(mse_asap):.4f}")
        print(f"[ASAP] Score range: {y_true_asap.min():.2f} - {y_true_asap.max():.2f}")
        print()
    
    commonlit_mask = sw_test["readability_output"] == 1
    if commonlit_mask.sum() > 0:
        y_true_commonlit = y_test["readability_output"][commonlit_mask]
        y_pred_commonlit = predictions[1][commonlit_mask].flatten()
        
        mse_commonlit = mean_squared_error(y_true_commonlit, y_pred_commonlit)
        mae_commonlit = np.mean(np.abs(y_true_commonlit - y_pred_commonlit))
        
        results["commonlit"] = {
            "mse": mse_commonlit,
            "mae": mae_commonlit,
            "rmse": np.sqrt(mse_commonlit),
            "n_samples": commonlit_mask.sum()
        }
        
        print(f"[CommonLit] Samples: {commonlit_mask.sum()}")
        print(f"[CommonLit] MSE: {mse_commonlit:.4f}")
        print(f"[CommonLit] MAE: {mae_commonlit:.4f}")
        print(f"[CommonLit] RMSE: {np.sqrt(mse_commonlit):.4f}")
        print(f"[CommonLit] Readability range: {y_true_commonlit.min():.2f} - {y_true_commonlit.max():.2f}")
        print()
    
    jfleg_mask = sw_test["jfleg_output"] == 1
    if jfleg_mask.sum() > 0:
        y_true_jfleg = y_test["jfleg_output"][jfleg_mask].astype(int)
        y_pred_jfleg_prob = predictions[2][jfleg_mask].flatten()
        y_pred_jfleg = (y_pred_jfleg_prob > 0.5).astype(int)
        
        accuracy = np.mean(y_pred_jfleg == y_true_jfleg)
        
        cm = confusion_matrix(y_true_jfleg, y_pred_jfleg)
        class_report = classification_report(y_true_jfleg, y_pred_jfleg, 
                                           target_names=['No Errors', 'Has Errors'])
        
        results["jfleg"] = {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "classification_report": class_report,
            "n_samples": jfleg_mask.sum(),
            "class_distribution": {
                "no_errors": (y_true_jfleg == 0).sum(),
                "has_errors": (y_true_jfleg == 1).sum()
            }
        }
        
        print(f"[JFLEG] Samples: {jfleg_mask.sum()}")
        print(f"[JFLEG] Accuracy: {accuracy:.4f}")
        print(f"[JFLEG] Class distribution:")
        print(f"  No errors (0): {(y_true_jfleg == 0).sum()}")
        print(f"  Has errors (1): {(y_true_jfleg == 1).sum()}")
        print()
        print("[JFLEG] Confusion Matrix:")
        print("          Predicted")
        print("        No Err  Has Err")
        print(f"No Err    {cm[0,0]:4d}    {cm[0,1]:4d}")
        print(f"Has Err   {cm[1,0]:4d}    {cm[1,1]:4d}")
        print()
        print("[JFLEG] Classification Report:")
        print(class_report)
    
    return results