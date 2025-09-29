# IDEAL-CT Automated App (Streamlit)

This repo hosts a Streamlit app for predicting IDEAL-CT from mix design inputs.

## Structure
```
Results/
  app.py
  BestModel_*.pkl            # optional if <100 MB; otherwise use MODEL_URL secret
requirements.txt
```

## Local run
```bash
cd Results
python -m streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push this folder to GitHub.
2. Go to https://streamlit.io/cloud → **New app** → pick your repo.
3. **Main file path:** `Results/app.py`
4. Deploy.

### If your .pkl > 100 MB
Do **not** commit it to GitHub. Host it (S3/Dropbox/Drive direct link) and set a secret:

**App → Settings → Secrets**:
```
MODEL_URL = "https://.../BestModel.pkl"
```
The app downloads the model at startup to `Results/BestModel_remote.pkl`.