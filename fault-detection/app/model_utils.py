"""Shared prediction code for the front end (kept free of Streamlit so it can be tested)."""
import sys
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from logistic import hypothesis  # noqa: E402

RES = ROOT / "results"


@lru_cache(maxsize=1)
def load_models():
    required = [RES / "theta_Q.npy", RES / "scaler_Q.npz", RES / "scaler_R.npz", RES / "svm_rbf_best.joblib"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {', '.join(missing)}")

    scaler_q = np.load(RES / "scaler_Q.npz")
    scaler_r = np.load(RES / "scaler_R.npz")
    theta_q = np.load(RES / "theta_Q.npy")
    try:
        svm = joblib.load(RES / "svm_rbf_best.joblib")
    except Exception as exc:  # pragma: no cover - surfaced to UI for clear error handling
        raise ValueError(f"Unable to load the saved RBF SVM model: {exc}") from exc

    return {
        "theta_Q": theta_q,
        "mu_Q": scaler_q["mu"], "sd_Q": scaler_q["sd"],
        "mu_R": scaler_r["mu"], "sd_R": scaler_r["sd"],
        "svm": svm,
    }


def _coerce_measurements(Ia, Ib, Ic, Va, Vb, Vc):
    values = np.asarray([Ia, Ib, Ic, Va, Vb, Vc], dtype=float)
    if values.shape != (6,):
        raise ValueError("Expected exactly six measurements.")
    if not np.all(np.isfinite(values)):
        raise ValueError("All measured values must be finite numbers.")
    return values


def logistic_set_q_probability(model_bundle, raw):
    q = np.concatenate([raw, raw ** 2])
    scaled_q = (q - model_bundle["mu_Q"]) / model_bundle["sd_Q"]
    xq = np.concatenate([[1.0], scaled_q])
    if not np.isfinite(xq).all():
        raise ValueError("Set Q feature preprocessing produced non-finite values.")
    p = float(hypothesis(xq.reshape(1, -1), model_bundle["theta_Q"])[0])
    if not np.isfinite(p):
        raise ValueError("Logistic probability is not finite.")
    return p


def svm_set_r_pred(model_bundle, raw):
    scaled_r = (raw - model_bundle["mu_R"]) / model_bundle["sd_R"]
    if not np.isfinite(scaled_r).all():
        raise ValueError("Set R feature preprocessing produced non-finite values.")
    y_svm = int(model_bundle["svm"].predict(scaled_r.reshape(1, -1))[0])
    return y_svm


def predict_both(m, Ia, Ib, Ic, Va, Vb, Vc):
    raw = _coerce_measurements(Ia, Ib, Ic, Va, Vb, Vc)
    p_lr = logistic_set_q_probability(m, raw)
    y_svm = svm_set_r_pred(m, raw)
    return {"lr_prob": p_lr, "lr_label": int(p_lr >= 0.5), "svm_label": y_svm}
