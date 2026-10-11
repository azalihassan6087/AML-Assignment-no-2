"""Task 7 - Professional Streamlit dashboard for transmission-line fault detection."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from model_utils import load_models, predict_both


# -------------------------------------------------------------------
# Configuration and paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_PATH = PROJECT_ROOT / "data" / "detect_dataset.csv"

st.set_page_config(
    page_title="GridGuard | Fault Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

FIGURES = {
    "Current signals and fault intervals": "fig1a_currents_time.png",
    "Current phase relationship": "fig1b_scatter_Ia_Ib.png",
    "Current distribution by class": "fig1c_hist_Ia.png",
    "Logistic regression learning curves": "fig2_cost_vs_iteration.png",
    "SVM validation comparison": "fig3_svm_accuracy_vs_C.png",
}


# -------------------------------------------------------------------
# Professional styling
# -------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #101d33;
    }

    [data-testid="stSidebar"] * {
        color: #eef4ff;
    }

    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #10233f, #164e63);
        color: white;
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
        font-weight: 750;
    }

    .hero p {
        color: #d8e8f6;
        margin-top: 0.6rem;
        margin-bottom: 0;
        font-size: 1rem;
    }

    .section-label {
        color: #52647b;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #52647b;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
    }

    .footnote {
        color: #65748b;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Cached data and model loading
# -------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def read_result_csv(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_dataset_summary():
    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH)
    df = df.drop(
        columns=[c for c in df.columns if df[c].isna().all()],
        errors="ignore",
    )

    label_col = "Output (S)" if "Output (S)" in df.columns else "fault"

    if label_col not in df.columns:
        return {"error": "Could not locate the dataset target column."}

    labels = pd.to_numeric(df[label_col], errors="coerce")

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "faults": int((labels == 1).sum()),
        "normal": int((labels == 0).sum()),
        "missing": int(df.isna().sum().sum()),
        "dataframe": df,
    }


@st.cache_resource(show_spinner="Loading trained models...")
def get_models():
    return load_models()


def show_csv(filename, title=None):
    """Display an existing result table without inventing values."""
    df = read_result_csv(filename)

    if df is None:
        st.warning(
            f"Missing results/{filename}. Run the relevant training "
            "or evaluation script to generate this artifact."
        )
        return None

    if title:
        st.subheader(title)

    st.dataframe(df, use_container_width=True, hide_index=True)
    return df


def show_figure(filename, caption=None):
    """Show a saved assignment figure, if available."""
    path = RESULTS_DIR / filename

    if not path.exists():
        st.warning(
            f"Figure not found: results/{filename}. "
            "Generate it using the existing EDA/training scripts."
        )
        return

    st.image(str(path), use_container_width=True)

    if caption:
        st.caption(caption)


def accuracy_as_percent(value):
    """Convert a numeric accuracy to a display percentage."""
    try:
        number = float(value)
        if 0 <= number <= 1:
            return f"{number:.2%}"
        return f"{number:.2f}%"
    except (ValueError, TypeError):
        return "N/A"


def first_matching_column(df, candidates):
    """Find a column despite minor capitalization differences."""
    if df is None:
        return None

    lookup = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


# -------------------------------------------------------------------
# Sidebar navigation
# -------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚡ GridGuard")
    st.caption("Transmission-line analytics")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Live Predictor",
            "Exploratory Analysis",
            "Training & Validation",
            "Test Evaluation",
            "Dataset & Methodology",
        ],
        label_visibility="visible",
    )

    st.divider()
    st.markdown("**AI4003 · Applied Machine Learning**")
    st.caption("Assignment 02 · Fall 2026")
    st.caption("Educational demonstration only")


# -------------------------------------------------------------------
# Hero section
# -------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>Three-Phase Transmission Line Fault Detection</h1>
        <p>
            An interactive machine-learning dashboard for analyzing
            current and voltage measurements, comparing classifiers,
            and reviewing experimental results.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Overview
# -------------------------------------------------------------------

if page == "Overview":
    st.markdown('<p class="section-label">PROJECT OVERVIEW</p>',
                unsafe_allow_html=True)
    st.subheader("Model performance at a glance")

    test_df = read_result_csv("table4_test_results.csv")
    prep_df = read_result_csv("table1_data_preparation.csv")

    if test_df is not None:
        accuracy_col = first_matching_column(test_df, ["Accuracy"])
        model_col = first_matching_column(test_df, ["Model"])
        missed_col = first_matching_column(
            test_df, ["Missed faults", "Missed Faults", "FN"]
        )
        false_col = first_matching_column(
            test_df, ["False alarms", "False Alarms", "FP"]
        )

        if accuracy_col and model_col:
            accuracies = pd.to_numeric(
                test_df[accuracy_col], errors="coerce"
            )
            best_idx = accuracies.idxmax()

            best_name = str(test_df.loc[best_idx, model_col])
            best_accuracy = accuracy_as_percent(
                test_df.loc[best_idx, accuracy_col]
            )

            missed_total = (
                int(pd.to_numeric(test_df[missed_col], errors="coerce").sum())
                if missed_col else None
            )
            false_total = (
                int(pd.to_numeric(test_df[false_col], errors="coerce").sum())
                if false_col else None
            )

            a, b, c, d = st.columns(4)

            a.metric("Models evaluated", len(test_df))
            b.metric("Best recorded accuracy", best_accuracy)
            c.metric(
                "Missed faults across models",
                f"{missed_total:,}" if missed_total is not None else "N/A",
            )
            d.metric(
                "False alarms across models",
                f"{false_total:,}" if false_total is not None else "N/A",
            )

            st.info(
                f"Highest recorded test accuracy: **{best_name}** "
                f"({best_accuracy}). These values come from the existing "
                "test-results table; the dashboard does not retrain models."
            )
            st.caption(
                "Missed faults and false alarms are summed across separate "
                "model evaluations. They are not counts for one deployed model."
            )
        else:
            st.warning("The test-results CSV has an unexpected column format.")
            st.dataframe(test_df, use_container_width=True)
    else:
        st.warning(
            "Test results are not available yet. Generate "
            "results/table4_test_results.csv first."
        )

    st.divider()
    st.subheader("Final test-set comparison")
    show_csv("table4_test_results.csv")

    st.subheader("Assignment's main question")
    st.markdown(
        """
        **Can a straight-line classifier detect faults using raw measurements?**

        Set R uses the six original measurements. Set Q adds squared
        features so logistic regression can learn a nonlinear relationship
        in the original measurement space. The RBF SVM instead uses a
        nonlinear kernel on the six standardized measurements.

        Compare validation results, missed faults, and false alarms before
        drawing conclusions about the most appropriate model.
        """
    )

    st.subheader("Explore the project")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Live prediction**")
        st.write("Enter six measurements and compare both saved models.")

    with c2:
        st.markdown("**Exploratory analysis**")
        st.write("Review the three assignment plots based on raw data.")

    with c3:
        st.markdown("**Model evaluation**")
        st.write("Inspect validation tables, confusion matrices, and test results.")


# -------------------------------------------------------------------
# Live predictor
# -------------------------------------------------------------------

elif page == "Live Predictor":
    st.markdown('<p class="section-label">LIVE INFERENCE</p>',
                unsafe_allow_html=True)
    st.subheader("Enter electrical measurements")
    st.write(
        "Provide one instant of the three-phase line currents and voltages. "
        "The saved Set Q logistic model and RBF SVM will make independent predictions."
    )

    example_choice = st.selectbox(
        "Example measurements",
        [
            "Custom values",
            "Normal operation — dataset example",
            "Fault condition — dataset example",
        ],
        key="example_choice",
    )

    # Load real rows only when the dataset is available.
    summary = read_dataset_summary()

    if example_choice != "Custom values":
        if summary is None or "dataframe" not in summary:
            st.warning(
                "The dataset is unavailable. You can still enter custom values."
            )
        else:
            df = summary["dataframe"]
            label_col = (
                "Output (S)" if "Output (S)" in df.columns else "fault"
            )

            target = 0 if example_choice.startswith("Normal") else 1
            label_values = pd.to_numeric(df[label_col], errors="coerce")
            rows = df.loc[label_values == target]

            if not rows.empty and all(c in rows.columns for c in FEATURES):
                # Stable sample: first valid row for the selected class.
                row = rows.iloc[0]
                for feature in FEATURES:
                    st.session_state[f"input_{feature}"] = float(row[feature])
            else:
                st.warning(
                    "A suitable example row was not found. Enter values manually."
                )

    with st.form("prediction_form"):
        st.markdown("#### Phase currents")
        current_cols = st.columns(3)

        current_values = []
        current_defaults = [0.0, 0.0, 0.0]

        for i, feature in enumerate(FEATURES[:3]):
            with current_cols[i]:
                value = st.number_input(
                    f"{feature} (A)",
                    value=float(
                        st.session_state.get(
                            f"input_{feature}", current_defaults[i]
                        )
                    ),
                    format="%.5f",
                    key=f"form_{feature}",
                    help="Instantaneous phase current in amperes.",
                )
                current_values.append(value)

        st.markdown("#### Phase voltages")
        voltage_cols = st.columns(3)
        voltage_values = []

        for i, feature in enumerate(FEATURES[3:]):
            with voltage_cols[i]:
                value = st.number_input(
                    f"{feature} (pu)",
                    value=float(
                        st.session_state.get(f"input_{feature}", 0.0)
                    ),
                    format="%.5f",
                    key=f"form_{feature}",
                    help="Instantaneous phase voltage in per unit.",
                )
                voltage_values.append(value)

        submitted = st.form_submit_button(
            "Analyze measurements",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        measurements = current_values + voltage_values

        if not np.isfinite(measurements).all():
            st.error("All six inputs must be finite numeric values.")
        else:
            try:
                model_bundle = get_models()

                with st.spinner("Evaluating saved models..."):
                    output = predict_both(model_bundle, *measurements)

                lr_label = int(output["lr_label"])
                svm_label = int(output["svm_label"])
                lr_probability = float(output["lr_prob"])

                if not np.isfinite(lr_probability) or not 0 <= lr_probability <= 1:
                    raise ValueError(
                        "The logistic model returned an invalid probability."
                    )

                st.session_state["latest_prediction"] = {
                    "measurements": measurements,
                    "output": output,
                }

            except Exception as exc:
                st.error(
                    "Prediction could not be completed. Verify that the "
                    "saved models, scaler files, and feature order are correct."
                )
                st.exception(exc)

    # Persist the most recent prediction across Streamlit reruns.
    latest = st.session_state.get("latest_prediction")

    if latest is not None:
        out = latest["output"]
        lr_label = int(out["lr_label"])
        svm_label = int(out["svm_label"])
        lr_probability = float(out["lr_prob"])

        st.divider()
        st.markdown('<p class="section-label">PREDICTION RESULTS</p>',
                    unsafe_allow_html=True)
        st.subheader("Classifier comparison")

        lr_col, svm_col = st.columns(2)

        with lr_col:
            st.markdown("### Logistic regression")
            st.caption("Set Q · 12 engineered features")

            if lr_label == 1:
                st.error("FAULT DETECTED")
            else:
                st.success("NO FAULT DETECTED")

            st.metric("Estimated probability of fault", f"{lr_probability:.2%}")
            st.progress(lr_probability)

        with svm_col:
            st.markdown("### RBF Support Vector Machine")
            st.caption("Set R · 6 standardized features")

            if svm_label == 1:
                st.error("FAULT DETECTED")
            else:
                st.success("NO FAULT DETECTED")

            st.info(
                f"Predicted class: {svm_label} — "
                f"{'Fault' if svm_label else 'No fault'}"
            )
            st.caption(
                "SVC class decision only. No calibrated probability is displayed."
            )

        if lr_label == svm_label:
            if lr_label == 1:
                st.warning(
                    "Both models predict a fault. This is an academic "
                    "prediction, not a validated relay trip command."
                )
            else:
                st.success("Both models predict no fault for this input.")
        else:
            st.warning(
                "The models disagree. Inspect the input measurements and "
                "evaluate the case using engineering validation."
            )

        with st.expander("Inspect submitted measurements"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "Measurement": FEATURES,
                        "Input value": latest["measurements"],
                        "Unit": ["A", "A", "A", "pu", "pu", "pu"],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.caption(
        "Model predictions are intended for academic demonstration only. "
        "They must not be used to operate real electrical protection equipment."
    )


# -------------------------------------------------------------------
# Exploratory analysis
# -------------------------------------------------------------------

elif page == "Exploratory Analysis":
    st.markdown('<p class="section-label">TASK 2 · EXPLORATORY DATA ANALYSIS</p>',
                unsafe_allow_html=True)
    st.subheader("Explore the raw measurement data")
    st.write(
        "These are the saved figures from the assignment's EDA pipeline. "
        "They use raw measurements rather than standardized features."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Current signals",
            "Current scatter",
            "Current distribution",
        ]
    )

    with tab1:
        show_figure(
            "fig1a_currents_time.png",
            "Ia, Ib and Ic against row number, with fault intervals shaded.",
        )
        st.markdown(
            """
            **Interpretation:** Compare the current traces and the shaded
            fault intervals. Look for changes in current magnitude and
            behavior around fault-labeled samples.
            """
        )

    with tab2:
        show_figure(
            "fig1b_scatter_Ia_Ib.png",
            "Scatter plot of Ia against Ib, colored by the fault label.",
        )
        st.markdown(
            """
            **Interpretation:** Examine the overlap between the two classes.
            Strong overlap can make a single straight-line boundary inadequate
            for separating fault and no-fault measurements.
            """
        )

    with tab3:
        show_figure(
            "fig1c_hist_Ia.png",
            "Overlapping distributions of Ia for fault and no-fault samples.",
        )
        st.markdown(
            """
            **Interpretation:** Compare the centers, spread and overlap of the
            two current distributions. A histogram of one feature alone does
            not establish whether all six measurements are separable.
            """
        )

    st.info(
        "For the written assignment, explain the actual patterns visible in "
        "each plot and relate them to the physical behavior of current during a fault."
    )


# -------------------------------------------------------------------
# Training and validation
# -------------------------------------------------------------------

elif page == "Training & Validation":
    st.markdown('<p class="section-label">TASKS 3–4 · MODEL DEVELOPMENT</p>',
                unsafe_allow_html=True)
    st.subheader("Logistic regression")

    show_csv("table2_logistic.csv", "Learning-rate selection and validation")

    show_figure(
        "fig2_cost_vs_iteration.png",
        "Cost J(theta) against iteration for the tested learning rates and feature sets.",
    )

    st.divider()
    st.subheader("Support vector machine validation")

    svm_df = show_csv(
        "table3_svm_validation.csv",
        "Linear and RBF SVM hyperparameter search",
    )

    if svm_df is not None:
        acc_col = first_matching_column(
            svm_df, ["Validation accuracy", "Validation Accuracy"]
        )
        if acc_col:
            st.caption(
                "The table reports validation accuracy and support-vector "
                "counts for the evaluated settings."
            )

    show_figure(
        "fig3_svm_accuracy_vs_C.png",
        "Validation accuracy against C on a logarithmic scale.",
    )

    params_path = RESULTS_DIR / "best_svm_params.json"
    if params_path.exists():
        with st.expander("Selected SVM parameters"):
            st.json(pd.read_json(params_path, typ="series").to_dict())
    else:
        st.warning("The selected SVM parameter file is not available.")


# -------------------------------------------------------------------
# Test evaluation and confusion matrices
# -------------------------------------------------------------------

elif page == "Test Evaluation":
    st.markdown('<p class="section-label">TASK 5 · FINAL EVALUATION</p>',
                unsafe_allow_html=True)
    st.subheader("Held-out test-set results")

    test_df = show_csv("table4_test_results.csv")

    st.markdown(
        """
        The confusion matrix convention used by this assignment is:

        - **TN:** normal sample correctly classified as normal.
        - **False alarm (FP):** normal sample classified as a fault.
        - **Missed fault (FN):** fault sample classified as normal.
        - **TP:** fault sample correctly classified as a fault.
        """
    )

    if test_df is not None:
        model_col = first_matching_column(test_df, ["Model"])
        tn_col = first_matching_column(test_df, ["TN"])
        fp_col = first_matching_column(
            test_df, ["False alarms", "False Alarms", "FP"]
        )
        fn_col = first_matching_column(
            test_df, ["Missed faults", "Missed Faults", "FN"]
        )
        tp_col = first_matching_column(test_df, ["TP"])

        if model_col and all([tn_col, fp_col, fn_col, tp_col]):
            st.subheader("Confusion matrices by model")

            selected_model = st.selectbox(
                "Choose a model",
                test_df[model_col].astype(str).tolist(),
            )

            row = test_df[
                test_df[model_col].astype(str) == selected_model
            ].iloc[0]

            tn = int(row[tn_col])
            fp = int(row[fp_col])
            fn = int(row[fn_col])
            tp = int(row[tp_col])

            matrix = pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=["Actual: No fault", "Actual: Fault"],
                columns=["Predicted: No fault", "Predicted: Fault"],
            )

            st.dataframe(
                matrix.style.format("{:,.0f}").background_gradient(
                    cmap="Blues", axis=None
                ),
                use_container_width=True,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("True negatives", f"{tn:,}")
            m2.metric("False alarms", f"{fp:,}")
            m3.metric("Missed faults", f"{fn:,}")
            m4.metric("True positives", f"{tp:,}")

            st.caption(
                f"Confusion-matrix total: {tn + fp + fn + tp:,} samples."
            )

            if fn > 0:
                st.warning(
                    f"This model missed {fn:,} fault samples in the recorded "
                    "test evaluation. Missed faults are particularly important "
                    "when discussing protection-system safety."
                )
            else:
                st.success("No missed faults are recorded for this model.")

            if tn + fp + fn + tp == 0:
                st.error("The confusion matrix is empty.")
        else:
            st.warning(
                "The test table does not contain the expected confusion-matrix columns."
            )


# -------------------------------------------------------------------
# Dataset and methodology
# -------------------------------------------------------------------

elif page == "Dataset & Methodology":
    st.markdown('<p class="section-label">TASK 1 · DATA PREPARATION</p>',
                unsafe_allow_html=True)
    st.subheader("Dataset overview")

    summary = read_dataset_summary()

    if summary is None:
        st.warning(
            "Dataset not found. Place detect_dataset.csv inside the data/ directory."
        )
    elif "error" in summary:
        st.error(summary["error"])
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw data rows", f"{summary['rows']:,}")
        c2.metric("Columns after empty-column removal", summary["columns"])
        c3.metric("Fault samples", f"{summary['faults']:,}")
        c4.metric("No-fault samples", f"{summary['normal']:,}")

        st.metric("Missing cells in loaded dataset", f"{summary['missing']:,}")

        with st.expander("Preview first five dataset rows"):
            st.dataframe(
                summary["dataframe"].head(),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "The summary above is calculated from the raw dataset with "
            "fully empty columns removed. It is not a substitute for the "
            "complete cleaning and split report."
        )

    st.divider()
    st.subheader("Assignment data-preparation report")
    show_csv("table1_data_preparation.csv")

    st.subheader("Feature engineering")

    left, right = st.columns(2)

    with left:
        st.markdown("### Set R")
        st.write("Six original measurements:")
        st.code("Ia, Ib, Ic, Va, Vb, Vc", language="text")

    with right:
        st.markdown("### Set Q")
        st.write("Six original measurements plus their six raw squares:")
        st.code(
            "Ia, Ib, Ic, Va, Vb, Vc,\n"
            "Ia², Ib², Ic², Va², Vb², Vc²",
            language="text",
        )

    st.subheader("Methodology")
    st.markdown(
        """
        1. Load and inspect the dataset.
        2. Apply the required random permutation using seed 42.
        3. Split into 60% training, 20% validation, and the remaining test data.
        4. Calculate standardization statistics from training data only.
        5. Train logistic regression manually with NumPy gradient descent.
        6. Evaluate linear and RBF SVM configurations using validation data.
        7. Select final models using validation results.
        8. Evaluate the selected models on the held-out test data.
        """
    )


# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------

st.divider()

st.caption(
    "GridGuard · AI4003 Applied Machine Learning · Assignment 02 · "
    "For educational use only."
)

st.caption(
    "Model accuracy on a randomly split dataset may overestimate "
    "performance on future transmission-line events. This dashboard "
    "is not certified protection equipment."
)