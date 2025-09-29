# app.py — IDEAL-CT Automated GUI (place this in Results/app.py)
import os, io, joblib, requests
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent         # .../Results
RESULTS_DIR = APP_DIR                             # models live here

# ---------- Page ----------
st.set_page_config(page_title="IDEAL-CT Automated Prediction App", layout="wide")

# ---------- Canonical headers + helpers ----------
CANON_FEATURES = ["NMAS", "Asphalt_Content", "RAP", "RAS", "VMA", "PG_High", "PG_Low"]
TARGET_CANDIDATES = ["IDEAL_CT", "ideal_ct", "target", "Target", "IDEALCT"]

ALIASES = {
    "id": "ID_code", "idcode": "ID_code", "id_code": "ID_code",
    "asphaltcontent": "Asphalt_Content", "asphalt_cont": "Asphalt_Content",
    "pg_high": "PG_High", "pg high": "PG_High",
    "pg_low": "PG_Low",  "pg low":  "PG_Low",
}
def _norm(s: str) -> str:
    return str(s).strip().replace("-", "_").replace(" ", "_").replace(".", "_").lower()

def normalize_dataframe(df: pd.DataFrame):
    # fix common header variants
    ren = {}
    for c in df.columns:
        n = _norm(c)
        if n in ALIASES: ren[c] = ALIASES[n]
        if n.startswith("asphalt_cont") and c != "Asphalt_Content":
            ren[c] = "Asphalt_Content"
    if ren: df = df.rename(columns=ren)

    target = None
    norm_targets = [t.lower() for t in TARGET_CANDIDATES]
    for c in df.columns:
        if _norm(c) in norm_targets:
            target = c
            break
    return df, target

def quick_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2)) if len(y_true) > 1 else np.nan
    r2 = 1 - ss_res/ss_tot if ss_tot and not np.isnan(ss_tot) else np.nan
    rmse = float(np.sqrt(np.mean(err**2))); mae = float(np.mean(np.abs(err)))
    return {"R2": r2, "RMSE": rmse, "MAE": mae}

# ---------- Model discovery / download ----------
def find_latest_model(results_dir: Path):
    cands = sorted(results_dir.glob("BestModel_*.pkl"), key=os.path.getmtime, reverse=True)
    return str(cands[0]) if cands else None

def ensure_model_file():
    """
    If no local BestModel_*.pkl found, try to download from secret MODEL_URL or env var.
    Saves to Results/BestModel_remote.pkl
    """
    local = find_latest_model(RESULTS_DIR)
    if local: return local
    url = st.secrets.get("MODEL_URL", "") or os.getenv("MODEL_URL", "")
    if not url:
        return None
    dest = RESULTS_DIR / "BestModel_remote.pkl"
    try:
        with st.status("Downloading model...", expanded=False):
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            dest.write_bytes(r.content)
        return str(dest)
    except Exception as e:
        st.error(f"Could not download model from URL. {e}")
        return None

@st.cache_resource(show_spinner=True)
def load_model(path: str):
    return joblib.load(path)

def get_model_features_fallback(model):
    # Prefer names from XGBoost booster; otherwise fall back to canonical
    try:
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            if booster is not None and getattr(booster, "feature_names", None):
                return list(booster.feature_names)
    except Exception:
        pass
    return CANON_FEATURES

# ---------- Sidebar: model ----------
st.sidebar.header("Model")
auto_model_path = ensure_model_file()
model_path_text = st.sidebar.text_input("Model path (.pkl)", value=str(auto_model_path or ""))
uploaded_model = st.sidebar.file_uploader("…or upload a .pkl", type=["pkl"])

model = None
model_features = CANON_FEATURES
if uploaded_model is not None:
    try:
        model = joblib.load(uploaded_model)
        model_features = get_model_features_fallback(model)
        st.sidebar.success("Loaded model from upload.")
    except Exception as e:
        st.sidebar.error(f"Could not load uploaded model:\n{e}")
elif model_path_text:
    try:
        model = load_model(model_path_text)
        model_features = get_model_features_fallback(model)
        st.sidebar.success(f"Loaded model: {Path(model_path_text).name}")
    except Exception as e:
        st.sidebar.error(f"Could not load model from path.\n{e}")
else:
    st.sidebar.info("Place BestModel_*.pkl in this Results folder, or set a MODEL_URL secret.")

st.sidebar.markdown("---")
st.sidebar.write("**Model expects features:**")
st.sidebar.code(", ".join(model_features), language="text")

# ---------- UI ----------
st.title("IDEAL-CT Automated Prediction App")
st.write("Use **Calculator** for a single mix, or **Batch** to score an Excel/CSV file.")
mode = st.radio("Choose mode:", ["Calculator (single mix)", "Batch (Excel/CSV)"], horizontal=True)

# ---------- Calculator ----------
if mode.startswith("Calculator"):
    if model is None:
        st.warning("Load or provide a model first (sidebar).")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        nmas = st.number_input("NMAS (mm)", min_value=4.75, max_value=37.5, value=12.5, step=0.25)
        rap  = st.number_input("RAP (%)",  min_value=0.0,  max_value=100.0, value=20.0, step=1.0)
    with c2:
        asphalt = st.number_input("Asphalt_Content (%)", min_value=2.0, max_value=8.0, value=5.0, step=0.1)
        ras     = st.number_input("RAS (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    with c3:
        vma  = st.number_input("VMA (%)", min_value=10.0, max_value=20.0, value=15.0, step=0.1)
        pg_h = st.number_input("PG_High (°C)", min_value=52, max_value=82, value=64, step=6)
    with c4:
        pg_l = st.number_input("PG_Low (°C)", min_value=-34, max_value=-10, value=-22, step=2)

    if st.button("Predict IDEAL-CT") and model is not None:
        row = pd.DataFrame({
            "NMAS": [nmas], "Asphalt_Content": [asphalt], "RAP": [rap],
            "RAS": [ras], "VMA": [vma], "PG_High": [pg_h], "PG_Low": [pg_l],
        }).reindex(columns=model_features, fill_value=0)
        try:
            yhat = float(model.predict(row)[0])
            st.success(f"**Predicted IDEAL-CT:** {yhat:.2f}")
            st.dataframe(row.assign(Predicted_IDEAL_CT=[yhat]))
        except Exception as e:
            st.error(f"Prediction failed:\n{e}")

# ---------- Batch ----------
else:
    if model is None:
        st.warning("Load or provide a model first (sidebar).")
    up = st.file_uploader("Upload NewData (.xlsx or .csv)", type=["xlsx", "csv"])
    left, right = st.columns(2)
    with left:
        infer_id = st.checkbox("Infer first column as ID if named like ID/ID_code", value=True)
    with right:
        show_metrics = st.checkbox("If IDEAL_CT exists, compute quick metrics", value=True)

    def template_bytes():
        df = pd.DataFrame(columns=["ID_code"] + CANON_FEATURES)
        df.loc[0] = ["mix_001", 12.5, 5.2, 20, 0, 15.5, 64, -22]
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="NewData")
        bio.seek(0); return bio
    st.download_button("Download NewData template (.xlsx)", data=template_bytes(),
                       file_name="NewData_Template.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if up is not None and model is not None:
        try:
            df = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        except Exception as e:
            st.error(f"Could not read file:\n{e}"); st.stop()

        df, target_col = normalize_dataframe(df)
        id_col = "ID_code" if "ID_code" in df.columns else None
        if infer_id and id_col is None:
            first = df.columns[0]
            if _norm(first) in ["id", "id_code", "idcode"]:
                df = df.rename(columns={first: "ID_code"}); id_col = "ID_code"

        missing = [c for c in CANON_FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.dataframe(pd.DataFrame({"Your columns": list(df.columns)}))
            st.stop()

        X = df[CANON_FEATURES].copy().reindex(columns=model_features, fill_value=0)
        try:
            preds = model.predict(X)
        except Exception as e:
            st.error(f"Prediction failed:\n{e}"); st.stop()

        out = df.copy(); out["Predicted_IDEAL_CT"] = preds
        st.subheader("Preview"); st.dataframe(out.head(50))

        if show_metrics and (target_col is not None) and (target_col in df.columns):
            try:
                m = quick_metrics(df[target_col].values, preds)
                st.info(f"**Quick metrics** → R²={m['R2']:.3f}, RMSE={m['RMSE']:.3f}, MAE={m['MAE']:.3f}")
            except Exception:
                pass

        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as w:
            out.to_excel(w, index=False, sheet_name="Predictions")
        bio.seek(0)
        st.download_button("Download predictions (.xlsx)", data=bio,
                           file_name="Predictions_Output.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------- Help ----------
with st.expander("Help & Notes"):
    st.markdown(f"""
- Models are searched in **{RESULTS_DIR}**. You can paste a path, upload a .pkl, or set **MODEL_URL** in Secrets.
- Header typos are auto-fixed (e.g., *Asphalt_Cont* → **Asphalt_Content**).
- **Calculator** predicts a single mix; **Batch** accepts .xlsx/.csv and exports predictions.
""")
