# models/shap_plot.py
import os, joblib, shap, pandas as pd, numpy as np, matplotlib.pyplot as plt

# ─── Chemins ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_FP   = os.path.join(BASE_DIR, "model_final.pkl")
PREP_FP    = os.path.join(BASE_DIR, "preprocessor.pkl")
XVALID_FP  = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "X_valid.csv"))

ASSET_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "assets"))
PNG_FP     = os.path.join(ASSET_DIR, "shap_global.png")
SIGNED_FP  = os.path.join(BASE_DIR, "shap_global_signed.csv")

os.makedirs(ASSET_DIR, exist_ok=True)

# ─── Chargement  & calcul SHAP ───────────────────────────────────
model    = joblib.load(MODEL_FP)
prep     = joblib.load(PREP_FP)
X_valid  = pd.read_csv(XVALID_FP)
X_proc   = prep.transform(X_valid)

explainer   = shap.TreeExplainer(model)
shap_vals   = explainer.shap_values(X_proc)
shap_vals   = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

# 1️⃣  PNG du SHAP global |SHAP| moyen
shap.summary_plot(shap_vals, X_valid, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(PNG_FP, dpi=150)
plt.close()

# 2️⃣  CSV des moyennes SHAP **signées**
mean_signed = shap_vals.mean(axis=0)           # signe conservé
pd.DataFrame({
    "feature": X_valid.columns,
    "mean_shap": mean_signed
}).to_csv(SIGNED_FP, index=False)

print("✅ SHAP global PNG et CSV générés")
