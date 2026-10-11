"""Task 3 - logistic regression by hand: choose alpha, train, report."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prepare import get_data, with_intercept, Q_NAMES, MEAS
from logistic import fit_logistic, predict, cost
from metrics import accuracy

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

ALPHAS = [0.01, 0.1, 1, 10]
N_ITERS = 2000
rows, thetas = [], {}
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, fs in zip(axes, ["R", "Q"]):
    d = get_data(fs)
    Xtr, ytr = with_intercept(d["X_train"]), d["y_train"]
    Xva, yva = with_intercept(d["X_val"]), d["y_val"]
    print(f"\n===== Set {fs}: {Xtr.shape[1]-1} features =====")
    print(f"J at theta=0: {cost(Xtr, ytr, np.zeros(Xtr.shape[1])):.6f} (should be log2 = 0.693147)")

    ok_alphas = []
    for a in ALPHAS:
        _, hist = fit_logistic(Xtr, ytr, a, N_ITERS)
        hist = np.array(hist)
        decreasing = bool(np.all(np.diff(hist) < 0))
        print(f"alpha={a:<5} final J={hist[-1]:.5f}  decreases every iteration: {decreasing}")
        ax.plot(hist, label=f"alpha={a}")
        if decreasing:
            ok_alphas.append(a)
    best = max(ok_alphas)                       # 3.4: largest alpha that always decreases
    print("chosen alpha:", best)
    ax.set_title(f"Set {fs}: J(theta) vs iteration")
    ax.set_xlabel("iteration"); ax.set_ylabel("J(theta)"); ax.set_yscale("log"); ax.legend()

    theta, hist = fit_logistic(Xtr, ytr, best, N_ITERS)       # 3.5 final model
    tr_acc = accuracy(ytr, predict(Xtr, theta))
    va_acc = accuracy(yva, predict(Xva, theta))
    print(f"train acc={tr_acc:.4f}  val acc={va_acc:.4f}  final J={hist[-1]:.5f}")
    names = ["theta0 (intercept)"] + (MEAS if fs == "R" else Q_NAMES)
    for n_, t in zip(names, theta):
        print(f"   {n_:20s} {t:+.4f}")

    thetas[fs] = theta
    rows.append({"Feature set": f"Set {fs}", "Chosen alpha": best,
                 "Final J(theta)": round(hist[-1], 5),
                 "Train accuracy": round(tr_acc, 4),
                 "Validation accuracy": round(va_acc, 4)})
    np.savez(RES / f"scaler_{fs}.npz", mu=d["mu"], sd=d["sd"])
    np.save(RES / f"theta_{fs}.npy", theta)
    pd.DataFrame({"weight": names, "theta": theta}).to_csv(
        RES / f"table5_theta_set{fs}.csv", index=False)

fig.tight_layout(); fig.savefig(RES / "fig2_cost_vs_iteration.png", dpi=150)
pd.DataFrame(rows).to_csv(RES / "table2_logistic.csv", index=False)
print("\nTable 2:\n", pd.DataFrame(rows).to_string(index=False))
