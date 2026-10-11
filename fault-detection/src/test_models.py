"""Task 5 - evaluate the four final models on the test rows, ONCE."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from prepare import get_data, with_intercept
from logistic import predict
from metrics import accuracy, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
params = json.loads((RES / "best_svm_params.json").read_text())

dR, dQ = get_data("R"), get_data("Q")
y = dR["y_test"]
assert (dR["y_test"] == dQ["y_test"]).all()

preds = {
    "Logistic regression, Set R": predict(with_intercept(dR["X_test"]), np.load(RES / "theta_R.npy")),
    "Logistic regression, Set Q": predict(with_intercept(dQ["X_test"]), np.load(RES / "theta_Q.npy")),
}
lin = joblib.load(RES / "svm_linear_best.joblib")
rbf = joblib.load(RES / "svm_rbf_best.joblib")
preds[f"Linear SVM (C={params['linear']['C']})"] = lin.predict(dR["X_test"])
preds[f"RBF SVM (C={params['rbf']['C']}, gamma={params['rbf']['gamma']})"] = rbf.predict(dR["X_test"])

rows = []
for name, p in preds.items():
    cm = confusion_matrix(y, p)
    assert cm.sum() == len(y)                              # correctness check
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    rows.append({"Model": name, "Accuracy": round(accuracy(y, p), 4),
                 "TN": tn, "False alarms": fp, "Missed faults": fn, "TP": tp})
    print(f"{name}\n  accuracy={accuracy(y, p):.4f}\n  confusion matrix [[TN, FP],[FN, TP]]:\n{cm}\n  missed faults={fn}\n")

table = pd.DataFrame(rows)
table.to_csv(RES / "table4_test_results.csv", index=False)
print("Table 4 (test rows =", len(y), "):\n", table.to_string(index=False))
