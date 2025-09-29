# Results/app.py
import io
import sys
from pathlib import Path
import requests
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="IDEAL-CT Automated Prediction App", layout="wide")

EXPECTED_COLS = ["NMAS", "Asphalt_Content", "RAP", "RAS", "VMA", "PG_High", "PG_Low"]

# -----------------------------
# Utilities
# -----------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # exact match required; allow harmless whitespace / case corrections
    mapping = {c.lower().strip(): c for c in df.columns}
    needed = [c.lower() for c in EXPECTED_COLS]
    if not all(x in mapping for x in needed):
        missing = [c for c in EXPECTED_COLS if c.lower() not in mapping]
        raise ValueError(f"Missing columns: {missing}")
    # reindex to exact order and names
    ordered = [mapping[c.lower()] for c in EXPECTED_COLS]
    out = df[ordered].copy()
    out.columns = EXPECTED_COLS
    return out

@st.cache_resource(show_spinner=False)
def load_model_auto(user_path: str | None) -> tuple[object, str]:
    """
    Load model in this precedence:
      1) First BestModel_*.pkl inside Results/ (repo)
      2) A user path (relative or absolute) provided in the sidebar
      3) Download from st.secrets['MODEL_URL'] (direct-download link)
      4) The file you uploaded in the sidebar (handled separately)
    Returns (model, how_loaded)
    """
    # 1) Repo model
    for p in sorted(Path("Results").glob("BestModel_*.pkl")):
        try:
            m = joblib.load(p)
            return m, f"repo:{p}"
        except Exception:
            pass  # try next

    # 2) User path
    if user_path:
        p = Path(user_path)
        if not p.exists():
            # try relative to repo root
            p = Path.cwd() / user_path.lstrip("/\\")
        if p.exists():
            m = joblib.load(p)
            return m, f"path:{p}"

    # 3) Secrets URL
    url = st.secrets.get("MODEL_URL", "").strip()
    if url:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        buf = io.BytesIO(r.content)
        m = joblib.load(buf)
        return m, "secrets:MODEL_URL"

    # 4) none found here (upload handled below)
    raise FileNotFoundError("No model found via repo, path, or MODEL_URL.")

def predict_df(model, X: pd.DataFrame) -> np.ndarray:
    # Order & numeric coercion
    Xn = _normalize_columns(X)
    for c in EXPECTED_COLS:
        Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    if Xn.isna().any().any():
        bad = Xn.columns[Xn.isna().any()].tolist()
        raise ValueError(f"Non-numeric or missing values in: {bad}")
    return np.asarray(model.predict(Xn), dtype=float)

# -----------------------------
# Sidebar: model loader
# -----------------------------
st.sidebar.header("Model")

path_box = st.sidebar.text_input(
    "Model path (.pkl)",
    value="Results/BestModel_RandomForest.pkl",
    help="Example: Results/BestModel_RandomForest.pkl (repo), or an absolute path."
)

uploaded = st.sidebar.file_uploader("...or upload a .pkl", type=["pkl"])

status_placeholder = st.sidebar.empty()

# Try auto sources
model = None
loaded_from = None
load_error = None
try:
    model, loaded_from = load_model_auto(path_box)
except Exception as e:
    load_error = str(e)

# If auto sources failed, try uploaded file
if model is None and uploaded is not None:
    try:
        model = joblib.load(uploaded)
        loaded_from = f"upload:{uploaded.name}"
        load_error = None
    except Exception as e:
        load_error = f"Upload error: {e}"

if model is None:
    status_placeholder.error(
        "No model loaded. Add a repo model under **Results/**, set a valid path, "
        "or add a direct-download link to `MODEL_URL` in **Settings → Secrets**."
    )
else:
    friendly = loaded_from.replace("repo:", "").replace("path:", "").replace("upload:", "")
    status_placeholder.success(f"Loaded model: {friendly}")

# -----------------------------
# Main UI
# -----------------------------
st.title("IDEAL-CT Automated Prediction App")
st.write("Use **Calculator** for a single mix, or **Batch** to score an Excel/CSV file.")

mode = st.radio("Choose mode:", ["Calculator (single mix)", "Batch (Excel/CSV)"], horizontal=True)

if model is None:
    st.info("⚠️ Load or provide a model first (sidebar).")
    st.stop()

if mode.startswith("Calculator"):
    cols = st.columns(7)
    vals = {}
    defaults = dict(NMAS=12.5, Asphalt_Content=5.0, RAP=20.0, RAS=0.0, VMA=15.0, PG_High=64, PG_Low=-22)
    ranges = dict(
        NMAS=(4.75, 19.0), Asphalt_Content=(4.0, 8.0), RAP=(0.0, 60.0), RAS=(0.0, 5.0),
        VMA=(10.0, 20.0), PG_High=(52, 82), PG_Low=(-34, -10)
    )
    for i, c in enumerate(EXPECTED_COLS):
        lo, hi = ranges[c]
        step = 0.01 if isinstance(lo, float) else 1
        vals[c] = cols[i].number_input(c.replace("_", " "), value=float(defaults[c]), min_value=float(lo), max_value=float(hi), step=step)

    if st.button("Predict IDEAL-CT", type="primary"):
        X = pd.DataFrame([{k: vals[k] for k in EXPECTED_COLS}])
        try:
            y = predict_df(model, X)
            st.success(f"Predicted IDEAL-CT: **{float(y[0]):.3f}**")
            st.dataframe(X.assign(Predicted_IDEAL_CT=y))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.write("Upload an **Excel (.xlsx)** or **CSV** with these exact columns:")
    st.code(", ".join(EXPECTED_COLS))
    colA, colB = st.columns([1,1])

    # Template download
    tmpl = pd.DataFrame([{c: "" for c in EXPECTED_COLS}])
    colA.download_button("Download template (.xlsx)", data=io.BytesIO(
        (lambda df: (df.to_excel(io.BytesIO(), index=False) or io.BytesIO()))(tmpl)
    ).getvalue(), file_name="IDEAL_CT_Template.xlsx")
    colB.download_button("Download template (.csv)", data=tmpl.to_csv(index=False).encode("utf-8"),
                         file_name="IDEAL_CT_Template.csv")

    up = st.file_uploader("Upload file", type=["xlsx", "csv"])
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                df = pd.read_csv(up)
            else:
                df = pd.read_excel(up)
            preds = predict_df(model, df)
            out = _normalize_columns(df).copy()
            out["Predicted_IDEAL_CT"] = preds
            st.success(f"Scored {len(out):,} rows.")
            st.dataframe(out.head(100))
            st.download_button("Download results (.xlsx)",
                               data=io.BytesIO((out.to_excel(io.BytesIO(), index=False) or io.BytesIO())).getvalue(),
                               file_name="IDEAL_CT_Predictions.xlsx")
        except Exception as e:
            st.error(f"Batch scoring failed: {e}")

with st.expander("Help & Notes"):
    st.markdown("""
- **Permanent model:**  
  1) Commit `Results/BestModel_*.pkl` to your repo (recommended), or  
  2) Paste a direct download link in **Settings → Secrets** as:
     ```toml
     MODEL_URL = "https://.../BestModel_RandomForest.pkl?download=1"
     ```
- **Required columns:** `NMAS, Asphalt_Content, RAP, RAS, VMA, PG_High, PG_Low`  
  Names must match exactly (case/spacing normalized).
- **Troubleshooting:** If a value can’t be parsed to a number, you’ll get a clear error listing the bad column(s).
""")
