# models/shap_global.py
import os
import joblib
import shap
import pandas as pd
import plotly.express as px
from functools import lru_cache

# ─────────────── Chemins relatifs ────────────────
BASE_DIR        = os.path.dirname(__file__)
MODEL_PATH      = os.path.join(BASE_DIR, "model_final.pkl")
PREPROC_PATH    = os.path.join(BASE_DIR, "preprocessor.pkl")
XTRAIN_PATH     = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "X_valid.csv"))

# ─────────────── SHAP Global Plot ────────────────
@lru_cache(maxsize=1)
def get_plotly_fig(top_k: int = 15):
    # Chargement des données et objets
    model       = joblib.load(MODEL_PATH)
    preproc     = joblib.load(PREPROC_PATH)
    X_train     = pd.read_csv(XTRAIN_PATH)
    X_proc      = preproc.transform(X_train)

    # Explainer LightGBM natif
    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(X_proc)
    shap_vals   = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    # Calcul |SHAP| moyen
    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": abs(shap_vals).mean(axis=0)
    }).sort_values("importance", ascending=False).head(top_k)

    # Plot interactif
    fig = px.bar(
        importance_df[::-1],  # inverse pour avoir le plus important en haut
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Blues",
        labels={"importance": "|SHAP| moyen", "feature": "Variable"},
        title=f"Top {top_k} variables influentes – SHAP global"
    )
    fig.update_layout(yaxis=dict(title=""), template="plotly_dark")
    return fig
