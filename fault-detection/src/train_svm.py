"""Task 4 - linear and RBF SVMs (sklearn.svm.SVC only) on Set R."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from prepare import get_data
from metrics import accuracy

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)


def deterministic_key_for_model(candidate, model_accuracy):
    """Validation accuracy decides the winner; ties use smaller C and then smaller gamma."""
    _, c, gamma = candidate
    return (-model_accuracy, c, 0.0 if gamma is None else float(gamma))


d = get_data("R")
Xtr, ytr, Xva, yva = d["X_train"], d["y_train"], d["X_val"], d["y_val"]
Cs, gammas = [0.1, 1, 10, 100], [0.1, 1]

rows, models = [], {}
for C in Cs:                                              # 4.1
    m = SVC(kernel="linear", C=C).fit(Xtr, ytr)
    acc = accuracy(yva, m.predict(Xva))
    rows.append({"Kernel": "linear", "C": C, "gamma": "-",
                 "Validation accuracy": round(acc, 4), "Support vectors": int(m.n_support_.sum())})
    models[("linear", C, None)] = (m, acc)
    print(rows[-1])
for C in Cs:                                              # 4.2
    for g in gammas:
        m = SVC(kernel="rbf", C=C, gamma=g).fit(Xtr, ytr)
        acc = accuracy(yva, m.predict(Xva))
        rows.append({"Kernel": "rbf", "C": C, "gamma": g,
                     "Validation accuracy": round(acc, 4), "Support vectors": int(m.n_support_.sum())})
        models[("rbf", C, g)] = (m, acc)
        print(rows[-1])

table = pd.DataFrame(rows)                                # 4.3
table.to_csv(RES / "table3_svm_validation.csv", index=False)
print("\nTable 3:\n", table.to_string(index=False))

best = {}                                                 # 4.4
for kern in ["linear", "rbf"]:
    candidates = [k for k in models if k[0] == kern]
    best_key = sorted(candidates, key=lambda k: deterministic_key_for_model(k, models[k][1]))[0]
    best[kern] = best_key
    joblib.dump(models[best_key][0], RES / f"svm_{kern}_best.joblib")
    print(f"best {kern}: C={best_key[1]}, gamma={best_key[2]}, val acc={models[best_key][1]:.4f}")
pd.Series({k: {"C": v[1], "gamma": v[2]} for k, v in best.items()}).to_json(RES / "best_svm_params.json")

# Figure 3: validation accuracy vs C (log scale)
fig, ax = plt.subplots(figsize=(6, 4))
lin = table[table.Kernel == "linear"]
ax.plot(lin.C, lin["Validation accuracy"], "o-", label="linear")
for g in gammas:
    r = table[(table.Kernel == "rbf") & (table.gamma == g)]
    ax.plot(r.C, r["Validation accuracy"], "s-", label=f"rbf, gamma={g}")
ax.set_xscale("log"); ax.set_xlabel("C"); ax.set_ylabel("validation accuracy")
ax.set_title("SVM validation accuracy vs C"); ax.legend()
fig.tight_layout(); fig.savefig(RES / "fig3_svm_accuracy_vs_C.png", dpi=150)
