"""Task 2 - exploratory plots from the RAW (unstandardised) data."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load_data import load_raw
from prepare import clean

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

df, _ = clean(load_raw(ROOT / "data"))

# 2.1 currents vs row number, first 2000 rows, fault shaded
sub = df.iloc[:2000]
fig, ax = plt.subplots(figsize=(12, 4))
for c, col in zip(["Ia", "Ib", "Ic"], ["tab:red", "tab:green", "tab:blue"]):
    ax.plot(sub.index, sub[c], label=c, lw=0.8, color=col)
f = sub["fault"].to_numpy()
start = None
for i, v in enumerate(f):
    if v == 1 and start is None:
        start = i
    if start is not None and (v == 0 or i == len(f) - 1):
        end = i if v == 0 else i + 1
        ax.axvspan(start, end, color="orange", alpha=0.25, lw=0)
        start = None
ax.set_xlabel("row number (time sample)")
ax.set_ylabel("current (A)")
ax.set_title("Line currents, first 2000 rows (orange = fault)")
ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(RES / "fig1a_currents_time.png", dpi=150); plt.close(fig)

# 2.2 scatter Ia vs Ib
fig, ax = plt.subplots(figsize=(6, 5))
for lab, col, name in [(0, "tab:blue", "no fault"), (1, "tab:red", "fault")]:
    m = df.fault == lab
    ax.scatter(df.loc[m, "Ia"], df.loc[m, "Ib"], s=4, alpha=0.4, c=col, label=name)
ax.set_xlabel("Ia (A)"); ax.set_ylabel("Ib (A)")
ax.set_title("Ia vs Ib coloured by class"); ax.legend(markerscale=3)
fig.tight_layout(); fig.savefig(RES / "fig1b_scatter_Ia_Ib.png", dpi=150); plt.close(fig)

# 2.3 histogram of Ia
fig, ax = plt.subplots(figsize=(6, 4))
bins = np.linspace(df.Ia.min(), df.Ia.max(), 80)
ax.hist(df.loc[df.fault == 0, "Ia"], bins=bins, alpha=0.6, color="tab:blue", label="no fault")
ax.hist(df.loc[df.fault == 1, "Ia"], bins=bins, alpha=0.6, color="tab:red", label="fault")
ax.set_xlabel("Ia (A)"); ax.set_ylabel("count"); ax.set_title("Histogram of Ia")
ax.legend()
fig.tight_layout(); fig.savefig(RES / "fig1c_hist_Ia.png", dpi=150); plt.close(fig)

print("Ia summary by class:")
print(df.groupby("fault")["Ia"].describe().round(1).to_string())
print("\n|Ia| summary by class:")
print(df.assign(absIa=df.Ia.abs()).groupby("fault")["absIa"].describe().round(1).to_string())
print("\nfraction of no-fault rows with |Ia| > 200:",
      round((df.loc[df.fault == 0, "Ia"].abs() > 200).mean(), 3))
print("fraction of fault rows with |Ia| < 200:",
      round((df.loc[df.fault == 1, "Ia"].abs() < 200).mean(), 3))
print("Figures saved to results/")
