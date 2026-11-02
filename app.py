from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DEFAULT_THRESHOLD = 0.40

st.set_page_config(
    page_title="Churn intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #edf0ff;
        --muted: #9ba6c5;
        --panel: #141c32;
        --line: rgba(159, 174, 218, .16);
        --accent: #9b8cff;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 75% -15%, rgba(116, 95, 231, .20), transparent 38%),
            #0b1020;
        color: var(--ink);
    }
    [data-testid="stSidebar"] {
        background: #10172a;
        border-right: 1px solid var(--line);
    }
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 16px 18px;
    }
    [data-testid="stMetricLabel"] p { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    h1, h2, h3 { letter-spacing: -.025em; }
    .eyebrow {
        color: #b7adff;
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .subtle { color: var(--muted); }
    div[data-testid="stFileUploader"] section {
        border: 1px dashed rgba(159, 174, 218, .35);
        border-radius: 14px;
        background: rgba(20, 28, 50, .55);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _clean_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"CSV is missing required customer fields: {', '.join(missing)}")

    features = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    features[NUMERIC_FEATURES] = features[NUMERIC_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    for column in CATEGORICAL_FEATURES:
        values = features[column].astype("string").str.strip()
        values = values.replace(
            {"No internet service": "No", "No phone service": "No"}
        )
        normalized = values.str.lower()
        values = values.mask(normalized.eq("yes"), "Yes")
        values = values.mask(normalized.eq("no"), "No")
        values = values.mask(normalized.eq("male"), "Male")
        values = values.mask(normalized.eq("female"), "Female")
        features[column] = values

    return features


def prepare_training_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, int]:
    if "Churn" not in frame.columns:
        raise ValueError("Training CSV must include a 'Churn' column with Yes/No labels.")

    labels = (
        frame["Churn"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0, "1": 1, "0": 0})
    )
    valid_rows = labels.notna()
    dropped_rows = int((~valid_rows).sum())
    if not valid_rows.any():
        raise ValueError("No usable Churn labels found. Labels must be Yes or No.")

    features = _clean_feature_frame(frame.loc[valid_rows]).reset_index(drop=True)
    target = labels.loc[valid_rows].astype(int).reset_index(drop=True)
    labeled_data = features.copy()
    labeled_data["Churn"] = target
    return features, target, labeled_data, dropped_rows


def train_model(frame: pd.DataFrame) -> dict[str, Any]:
    features, target, labeled_data, dropped_rows = prepare_training_data(frame)
    class_counts = target.value_counts()
    if len(class_counts) != 2:
        raise ValueError("Training data must include both churn and non-churn customers.")
    if int(class_counts.min()) < 5:
        raise ValueError("Please provide at least 5 labeled examples in each Churn class.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "model": model,
        "x_test": x_test.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "probabilities": probabilities,
        "training_features": features,
        "labeled_data": labeled_data,
        "dropped_rows": dropped_rows,
        "row_count": len(features),
        "churn_rate": float(target.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def calculate_metrics(
    labels: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "flagged": int(predictions.sum()),
    }


def _style_chart(figure: go.Figure, height: int = 340) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=12, r=12, t=30, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#edf0ff"),
    )
    return figure


def _render_header() -> None:
    st.markdown('<p class="eyebrow">Customer retention · decision support</p>', unsafe_allow_html=True)
    st.title("Churn intelligence")
    st.markdown(
        '<p class="subtle">Find at-risk customers, understand the signals, and focus retention where it can make a difference.</p>',
        unsafe_allow_html=True,
    )


def _render_overview(bundle: dict[str, Any], metrics: dict[str, float | int]) -> None:
    st.subheader("Portfolio overview")
    st.caption(
        "Customer patterns below are descriptive. Model metrics are measured on a held-out test split."
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Labeled customers", f"{bundle['row_count']:,}")
    metric_columns[1].metric("Observed churn", f"{bundle['churn_rate']:.1%}")
    metric_columns[2].metric("Test recall", f"{metrics['recall']:.1%}")
    metric_columns[3].metric("Test ROC-AUC", f"{metrics['roc_auc']:.3f}")

    left, right = st.columns(2)
    churn_counts = (
        bundle["labeled_data"]["Churn"]
        .map({0: "Stayed", 1: "Churned"})
        .value_counts()
        .rename_axis("Customer outcome")
        .reset_index(name="Customers")
    )
    outcome_chart = px.pie(
        churn_counts,
        names="Customer outcome",
        values="Customers",
        hole=0.68,
        color="Customer outcome",
        color_discrete_map={"Stayed": "#7169d5", "Churned": "#f08c83"},
    )
    outcome_chart.update_traces(textposition="inside", textinfo="percent")
    left.markdown("#### Customer outcomes")
    left.plotly_chart(_style_chart(outcome_chart), use_container_width=True)

    contract_data = bundle["labeled_data"].copy()
    contract_data["Churn"] = contract_data["Churn"].astype(float)
    contract_rates = (
        contract_data.groupby("Contract", dropna=False, observed=True)["Churn"]
        .mean()
        .mul(100)
        .rename("Churn rate (%)")
        .reset_index()
        .sort_values("Churn rate (%)", ascending=False)
    )
    contract_chart = px.bar(
        contract_rates,
        x="Contract",
        y="Churn rate (%)",
        color="Churn rate (%)",
        color_continuous_scale=["#7771d7", "#efa08a"],
        text_auto=".1f",
    )
    contract_chart.update_layout(coloraxis_showscale=False)
    right.markdown("#### Churn rate by contract")
    right.plotly_chart(_style_chart(contract_chart), use_container_width=True)

    tenure_data = bundle["labeled_data"].copy()
    tenure_data["Tenure band"] = pd.cut(
        tenure_data["tenure"],
        bins=[-1, 12, 24, 36, 48, 60, np.inf],
        labels=["0-12 mo", "13-24 mo", "25-36 mo", "37-48 mo", "49-60 mo", "61+ mo"],
    )
    tenure_rates = (
        tenure_data.groupby("Tenure band", observed=True)["Churn"]
        .mean()
        .mul(100)
        .rename("Churn rate (%)")
        .reset_index()
    )
    tenure_chart = px.line(
        tenure_rates,
        x="Tenure band",
        y="Churn rate (%)",
        markers=True,
        color_discrete_sequence=["#a99cff"],
    )
    tenure_chart.update_yaxes(rangemode="tozero")
    st.markdown("#### Churn across customer tenure")
    st.plotly_chart(_style_chart(tenure_chart, 310), use_container_width=True)


def _render_model_performance(
    bundle: dict[str, Any],
    threshold: float,
    metrics: dict[str, float | int],
) -> None:
    st.subheader("Held-out model performance")
    st.caption(
        f"Threshold {threshold:.2f} · {len(bundle['y_test']):,} test customers · "
        "the test split was not used to fit preprocessing or the model."
    )
    cards = st.columns(4)
    cards[0].metric("Recall", f"{metrics['recall']:.1%}")
    cards[1].metric("Precision", f"{metrics['precision']:.1%}")
    cards[2].metric("F1 score", f"{metrics['f1']:.3f}")
    cards[3].metric("Flagged in test set", f"{metrics['flagged']:,}")

    predictions = (
        bundle["probabilities"] >= threshold
    ).astype(int)
    matrix = confusion_matrix(bundle["y_test"], predictions, labels=[0, 1])
    matrix_chart = px.imshow(
        matrix,
        text_auto=True,
        x=["Predicted stayed", "Predicted churn"],
        y=["Actually stayed", "Actually churned"],
        color_continuous_scale=["#18213b", "#8c7cff"],
        labels={"x": "", "y": "", "color": "Customers"},
        aspect="auto",
    )
    st.plotly_chart(_style_chart(matrix_chart, 290), use_container_width=True)
    st.caption(
        "Reducing the threshold usually catches more churners, but can also flag "
        "more customers who would have stayed."
    )


def _render_single_customer(bundle: dict[str, Any], threshold: float) -> None:
    st.subheader("Score one customer")
    st.caption("Enter the customer’s current account and service details.")
    training_features = bundle["training_features"]
    customer: dict[str, Any] = {}

    sections = {
        "Customer profile": ["gender", "SeniorCitizen", "Partner", "Dependents"],
        "Account and billing": [
            "tenure",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
        ],
        "Services": [
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ],
    }

    with st.form("customer_score_form"):
        for section, fields in sections.items():
            st.markdown(f"#### {section}")
            columns = st.columns(2)
            for index, field in enumerate(fields):
                with columns[index % 2]:
                    if field in NUMERIC_FEATURES:
                        values = pd.to_numeric(training_features[field], errors="coerce")
                        default = float(values.median()) if values.notna().any() else 0.0
                        if field == "SeniorCitizen":
                            customer[field] = st.selectbox(
                                "Senior citizen",
                                options=[0, 1],
                                format_func=lambda value: "Yes" if value else "No",
                            )
                        elif field == "tenure":
                            customer[field] = st.number_input(
                                "Tenure (months)",
                                min_value=0,
                                max_value=1000,
                                value=int(round(default)),
                                step=1,
                            )
                        else:
                            customer[field] = st.number_input(
                                "Monthly charges",
                                min_value=0.0,
                                max_value=10000.0,
                                value=round(default, 2),
                                step=1.0,
                            )
                    else:
                        options = (
                            training_features[field]
                            .dropna()
                            .astype(str)
                            .value_counts()
                            .index.tolist()
                        )
                        if not options:
                            options = ["Unknown"]
                        customer[field] = st.selectbox(field, options=options)
        submitted = st.form_submit_button(
            "Estimate churn risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        customer_frame = _clean_feature_frame(pd.DataFrame([customer]))
        probability = float(bundle["model"].predict_proba(customer_frame)[0, 1])
        predicted_churn = probability >= threshold
        st.markdown("---")
        result_columns = st.columns([1, 2])
        result_columns[0].metric("Estimated churn probability", f"{probability:.1%}")
        if predicted_churn:
            result_columns[1].error(
                "Above the current alert threshold. Consider a timely, relevant retention conversation."
            )
        else:
            result_columns[1].success(
                "Below the current alert threshold. Continue standard customer care."
            )
        st.caption(
            "This is a model estimate, not a certainty or a causal explanation. "
            "Use it alongside customer context."
        )


def _render_bulk_scoring(bundle: dict[str, Any], threshold: float) -> None:
    st.subheader("Score a customer list")
    st.caption(
        "Upload a CSV with the same customer fields used for training. "
        "The file is processed in this session and is not saved by the app."
    )
    scoring_file = st.file_uploader(
        "Customer CSV",
        type=["csv"],
        key="scoring_csv",
        help="A Churn column is not required for scoring.",
    )
    if scoring_file is None:
        return

    try:
        uploaded = pd.read_csv(BytesIO(scoring_file.getvalue()))
        features = _clean_feature_frame(uploaded)
        probabilities = bundle["model"].predict_proba(features)[:, 1]
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not score this CSV: {error}")
        return

    results = uploaded.copy()
    results["Churn probability"] = probabilities
    results["Predicted churn"] = np.where(probabilities >= threshold, "Yes", "No")
    results["Retention priority"] = np.where(
        probabilities >= threshold, "Review", "Standard"
    )
    flagged = int((probabilities >= threshold).sum())
    summary = st.columns(3)
    summary[0].metric("Customers scored", f"{len(results):,}")
    summary[1].metric("Flagged for review", f"{flagged:,}")
    summary[2].metric(
        "Flagged share",
        f"{flagged / len(results):.1%}" if len(results) else "0.0%",
    )
    st.dataframe(
        results.sort_values("Churn probability", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download scored CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="churn_risk_scores.csv",
        mime="text/csv",
        type="primary",
    )


def _render_model_insights(bundle: dict[str, Any]) -> None:
    st.subheader("What the model learned")
    st.caption(
        "Relative model associations, not causes. Numeric features are standardized; "
        "categorical features are one-hot encoded."
    )
    preprocessing = bundle["model"].named_steps["preprocessing"]
    categorical_encoder = (
        preprocessing.named_transformers_["categorical"].named_steps["onehot"]
    )
    feature_names = NUMERIC_FEATURES + categorical_encoder.get_feature_names_out(
        CATEGORICAL_FEATURES
    ).tolist()
    coefficients = bundle["model"].named_steps["classifier"].coef_[0]
    importance = pd.DataFrame(
        {"Feature": feature_names, "Model coefficient": coefficients}
    )
    importance["Magnitude"] = importance["Model coefficient"].abs()
    importance = importance.nlargest(18, "Magnitude").sort_values(
        "Model coefficient"
    )
    importance["Direction"] = np.where(
        importance["Model coefficient"] >= 0,
        "Higher model score",
        "Lower model score",
    )
    chart = px.bar(
        importance,
        x="Model coefficient",
        y="Feature",
        color="Direction",
        orientation="h",
        color_discrete_map={
            "Higher model score": "#f08c83",
            "Lower model score": "#8f86ed",
        },
    )
    chart.add_vline(x=0, line_color="rgba(255,255,255,.35)")
    st.plotly_chart(_style_chart(chart, 500), use_container_width=True)


def _render_empty_state() -> None:
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("### Start with your customer data")
        st.markdown(
            "Upload the Telco Customer Churn CSV, train an explainable baseline, "
            "and explore customer-level or portfolio-level risk."
        )
        st.markdown(
            "**Required columns:** `Churn`, `gender`, `SeniorCitizen`, `Partner`, "
            "`Dependents`, `tenure`, `PhoneService`, `MultipleLines`, "
            "`InternetService`, `OnlineSecurity`, `OnlineBackup`, "
            "`DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, "
            "`Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`."
        )
        st.markdown(
            "The original Telco dataset is available from "
            "[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)."
        )
    with right:
        st.info(
            "No dataset or pre-trained model is bundled with this project. "
            "Upload a labeled CSV in the sidebar to train the model. "
            "The `customerID` and `TotalCharges` columns are optional and are not "
            "used as model features."
        )


def main() -> None:
    _render_header()
    st.sidebar.markdown("## Data & model")
    st.sidebar.caption("Train an explainable churn model from a labeled customer CSV.")
    training_file = st.sidebar.file_uploader(
        "Training CSV",
        type=["csv"],
        key="training_csv",
        help="The file must include Churn labels and the Telco customer feature columns.",
    )

    uploaded_frame: pd.DataFrame | None = None
    fingerprint: str | None = None
    if training_file is not None:
        content = training_file.getvalue()
        fingerprint = hashlib.sha256(content).hexdigest()
        try:
            uploaded_frame = pd.read_csv(BytesIO(content))
        except (pd.errors.ParserError, UnicodeDecodeError) as error:
            st.sidebar.error(f"Could not read the CSV: {error}")

    trained_fingerprint = st.session_state.get("trained_fingerprint")
    if (
        trained_fingerprint is not None
        and fingerprint is not None
        and trained_fingerprint != fingerprint
    ):
        st.session_state.pop("model_bundle", None)
        st.session_state.pop("trained_fingerprint", None)

    can_train = uploaded_frame is not None
    if st.sidebar.button(
        "Train / refresh model",
        type="primary",
        use_container_width=True,
        disabled=not can_train,
    ):
        try:
            with st.spinner("Training and evaluating on a held-out test split..."):
                st.session_state["model_bundle"] = train_model(uploaded_frame)
                st.session_state["trained_fingerprint"] = fingerprint
        except (ValueError, KeyError) as error:
            st.session_state.pop("model_bundle", None)
            st.session_state.pop("trained_fingerprint", None)
            st.sidebar.error(str(error))

    bundle = st.session_state.get("model_bundle")
    if bundle is None:
        _render_empty_state()
        return

    st.sidebar.success(
        f"Model ready · {bundle['row_count']:,} labeled customers"
    )
    if bundle["dropped_rows"]:
        st.sidebar.warning(
            f"Ignored {bundle['dropped_rows']:,} rows without a recognized Churn label."
        )
    threshold = st.sidebar.slider(
        "Churn alert threshold",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help="Lower thresholds flag more customers and usually increase recall.",
    )
    metrics = calculate_metrics(
        bundle["y_test"], bundle["probabilities"], threshold
    )

    tabs = st.tabs(
        ["Overview", "Score a customer", "Bulk scoring", "Model insights"]
    )
    with tabs[0]:
        _render_overview(bundle, metrics)
        _render_model_performance(bundle, threshold, metrics)
    with tabs[1]:
        _render_single_customer(bundle, threshold)
    with tabs[2]:
        _render_bulk_scoring(bundle, threshold)
    with tabs[3]:
        _render_model_insights(bundle)

    st.sidebar.caption(
        "Preprocessing is fitted on training data only. Customer identifiers and "
        "TotalCharges are excluded from model features."
    )


if __name__ == "__main__":
    main()
