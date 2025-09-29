# Results/app.py
# IDEAL-CT Automated Prediction App (Streamlit)
# - Loads a trained .pkl model (local file in /Results or via secret URL)
# - Single-mix "Calculator" and Batch (Excel/CSV) prediction
# - FIX: number_input uses all-float args (value/min/max/step) to avoid type mismatch

import os
import io
import json
import time
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# -----------------------------
# Config / defaults
# -----------------------------
st.set_page_config(page_title="IDEAL-CT Automated Prediction App", layout="wide")

# Expected raw feature order (update here if your model expects more/less/different)
EXPECTED_COLS: List[str] = [
    "NMAS", "Asphalt_Content", "RAP", "RAS", "VMA", "PG_High", "PG_Low"
]

# Default local model path inside the repo (you can rename if needed)
DEFAULT_MODEL_PATH = "/mount/src/idealct-app/Results/BestModel_remote.pkl"

# Default UI ranges (ALL floats to keep number_input consistent)
RANGES = dict(
    NMAS=(4.75, 19.0),
    Asphalt_Content=(3.0, 9.0),
    RAP=(0.0, 60.0),
    RAS=(0.0, 5.0),
    VMA=(10.0, 20.0),
    PG_High=(52.0, 82.0),
    PG_Low=(-34.0, -10.0),
)

DEFAULTS = dict(
    NMAS=12.5, Asphalt_Content=5.0, RAP=20.0, RAS=0.0, VMA=15.0, PG_High=64.0, PG_Low=-22.0
)


# -----------------------------
# Utility helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def _download_bytes(url: str) -> bytes:
    """Download bytes from a URL (OneDrive 'download=1' or any direct link)."""
    import requests
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


@st.cache_resource(show_spinner=False)
def load_model_from_bytes(b: bytes):
    """Load a Joblib model from raw bytes (cached)."""
    buf = io.BytesIO(b)
    model = joblib.load(buf)
    return model


@st.cache_resource(show_spinner=False)
def load_model_from_path(path: str):
    """Load a Joblib model from a local path (cached)."""
    model = joblib.load(path)
    return model


def get_model():
    """
    Try to load model in this order:
    1) Uploaded .pkl file (left sidebar)
    2) Secret URL: st.secrets["MODEL_URL"] (must be a DIRECT download link)
    3) Local file DEFAULT_MODEL_PATH
    """
    # 1) Uploaded file?
    up = st.session_state.get("_uploaded_model_file", None)
    if up is not None:
        try:
            return load_model_from_bytes(up.getvalue()), "Uploaded file"
        except Exception as e:
            st.warning(f"Could not load uploaded model: {e}")

    # 2) Secret URL?
    model_url = None
    try:
        model_url = st.secrets.get("MODEL_URL", None)
    except Exception:
        model_url = None

    if model_url:
        try:
            b = _download_bytes(model_url)
            model = load_model_from_bytes(b)
            return model, "Secret URL"
        except Exception as e:
            st.error(f"Could not download model from URL.\n{e}")

    # 3) Local file
    local_path = st.session_state.get("_model_path_text", DEFAULT_MODEL_PATH)
    if local_path and os.path.exists(local_path):
        try:
            return load_model_from_path(local_path), local_path
        except Exception as e:
            st.error(f"Could not load model from path.\n{e}")
    else:
        st.info("No valid model found yet. Provide a .pkl in the sidebar.")


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to numeric if possible."""
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def prep_features(df: pd.DataFrame, expected=EXPECTED_COLS) -> pd.DataFrame:
    """Reindex to expected columns and fill missing with 0."""
    df = df.copy()
    df = ensure_numeric(df)
    df = df.reindex(columns=expected, fill_value=0.0)
    return df


def predict_df(model, df_features: pd.DataFrame) -> np.ndarray:
    X = prep_features(df_features, EXPECTED_COLS)
    y = model.predict(X)
    return np.asarray(y).ravel()


# -----------------------------
# Sidebar (model loading)
# -----------------------------
with st.sidebar:
    st.subheader("Model")
    # Text path (local)
    path_text = st.text_input(
        "Model path (.pkl)",
        value=st.session_state.get("_model_path_text", DEFAULT_MODEL_PATH),
    )
    st.session_state["_model_path_text"] = path_text

    st.caption("…or upload a .pkl")
    upl = st.file_uploader(
        "Drag and drop file here",
        type=["pkl"],
        label_visibility="collapsed"
    )
    if upl is not None:
        st.session_state["_uploaded_model_file"] = upl
        st.success("Loaded model from upload.")

    st.info(
        "Place **BestModel_*.pkl** in this Results folder, or set a **MODEL_URL** secret "
        "(must be a direct download link)."
    )

    st.markdown("**Model expects features:**")
    st.code(", ".join(EXPECTED_COLS), language="text")


# -----------------------------
# Main UI
# -----------------------------
st.title("IDEAL-CT Automated Prediction App")

model_info = get_model()
if not model_info:
    st.stop()

model, loaded_from = model_info
st.success(f"Model loaded from: {loaded_from}")

st.write("Use **Calculator** for a single mix, or **Batch** to score an Excel/CSV file.")
mode = st.radio("Choose mode:", ["Calculator (single mix)", "Batch (Excel/CSV)"], horizontal=True)


# -----------------------------
# Calculator (single mix)
# -----------------------------
if mode.startswith("Calculator"):
    cols = st.columns(len(EXPECTED_COLS))
    vals = {}

    for i, c in enumerate(EXPECTED_COLS):
        lo, hi = RANGES[c]
        value = float(DEFAULTS[c])
        minv = float(lo)
        maxv = float(hi)
        # Use float step for all numeric inputs (avoid type mismatch)
        step = 0.01 if c not in ("PG_High", "PG_Low") else 1.0

        vals[c] = cols[i].number_input(
            c.replace("_", " "),
            value=value,
            min_value=minv,
            max_value=maxv,
            step=float(step)
        )

    if st.button("Predict IDEAL-CT", type="primary"):
        row = {k: float(vals[k]) for k in EXPECTED_COLS}
        X = pd.DataFrame([row])
        try:
            y = predict_df(model, X)
            st.success(f"Predicted IDEAL-CT: **{float(y[0]):.3f}**")
            st.dataframe(X.assign(Predicted_IDEAL_CT=y))
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# -----------------------------
# Batch scoring
# -----------------------------
else:
    st.write("Upload **Excel (.xlsx)** or **CSV** with columns:")
    st.code(", ".join(EXPECTED_COLS), language="text")

    f = st.file_uploader("Upload file", type=["xlsx", "csv"])
    if f is not None:
        try:
            if f.name.lower().endswith(".csv"):
                df_in = pd.read_csv(f)
            else:
                df_in = pd.read_excel(f)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        st.write("Preview:")
        st.dataframe(df_in.head())

        try:
            y = predict_df(model, df_in)
            out = df_in.copy()
            out["Predicted_IDEAL_CT"] = y
            st.success(f"Scored {len(out):,} rows.")
            st.dataframe(out.head(50))

            # Download button
            buf = io.BytesIO()
            if f.name.lower().endswith(".csv"):
                out.to_csv(buf, index=False)
                mime = "text/csv"
                fname = "Predictions.csv"
            else:
                with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
                    out.to_excel(w, index=False, sheet_name="Predictions")
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                fname = "Predictions.xlsx"

            st.download_button("Download results", buf.getvalue(), file_name=fname, mime=mime)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# -----------------------------
# Help & Notes
# -----------------------------
with st.expander("Help & Notes"):
    st.markdown(
        """
- If you see *"Could not download model from URL"* make sure your **MODEL_URL** is a direct download link
  (e.g., OneDrive link with `download=1`).
- To avoid number input errors, all inputs in the calculator use float `value/min/max/step`.
- Expected feature columns (case-sensitive):  
  `""" + ", ".join(EXPECTED_COLS) + "`"
    )
