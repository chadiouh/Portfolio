# ──────────────────────────────────────────────────────────────
# 4_Comparaison.py  –  Page Streamlit
# ----------------------------------------------------------------
# Affiche la position du client vs. population, avec coloration
# basée sur l’impact SHAP moyen local (favorable = vert, neutre =
# gris, défavorable = rouge).  Fonctionne quels que soient :
#   • l’emplacement du script  (pages/…)
#   • la plate-forme  (Windows, Linux, Render, …)
#   • le jeu de features conservé dans top_features.json
# ──────────────────────────────────────────────────────────────
import os, json, pickle, numpy as np, pandas as pd, plotly.express as px
import streamlit as st

# ────────────────────────── Paramètres globaux
NEUTRAL_EPS = 0.015                   # zone “impact neutre” (|SHAP| ≤ ε)

# ────────────────────────── Vérification d’état
if "user_input" not in st.session_state or "result" not in st.session_state:
    st.warning("⚠️ Veuillez d’abord effectuer une prédiction dans l’onglet *Formulaire*.")
    st.stop()

USER_INPUT = st.session_state["user_input"]

# ────────────────────────── Chemins robustes
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))          # …/app/pages
APP_DIR    = os.path.abspath(os.path.join(BASE_DIR, ".."))       # …/app
DATA_DIR   = os.path.join(APP_DIR, "data")
MODEL_DIR  = os.path.join(APP_DIR, "models")

FP_SAMPLE  = os.path.join(DATA_DIR,  "application_sample.csv")
FP_SHAP    = os.path.join(MODEL_DIR, "shap_summary_validation.pkl")
FP_TOPFEAT = os.path.join(MODEL_DIR, "top_features.json")

# ────────────────────────── Chargements avec cache
@st.cache_data
def load_everything():
    # Chemins des fichiers
    FP_SAMPLE = "data/application_sample.csv"
    FP_SHAP = "models/shap_summary_validation.pkl"
    FP_TOP_FEATS = "models/top_features.json"

    # Chargement des fichiers
    df_val = pd.read_csv(FP_SAMPLE)

    with open(FP_TOP_FEATS, "r") as f:
        TOP_FEATURES = json.load(f)

    with open(FP_SHAP, "rb") as f:
        shap_loc = pickle.load(f)

    # Correction : forcer en DataFrame si jamais shap_loc est une Series
    if isinstance(shap_loc, pd.Series):
        shap_loc = shap_loc.to_frame().T

    # On conserve uniquement les colonnes communes aux top_features et au shap
    valid_feats = [c for c in TOP_FEATURES if c in df_val.columns and c in shap_loc.columns]

    # Application du suffixe pour identifier les shap values
    shap_loc = shap_loc[valid_feats].add_suffix("_shap")

    # On concatène shap values et valeurs d'origine
    df_full = pd.concat([df_val[["SK_ID_CURR"] + valid_feats], shap_loc], axis=1)

    return df_full, valid_feats

df_full, VALID_FEATS = load_everything()

if not VALID_FEATS:
    st.error(
        "Aucune des variables conservées dans **top_features.json** "
        "n’apparaît à la fois dans *application_sample.csv* **et** dans *shap_summary_validation.pkl*."
    )
    st.stop()

# ────────────────────────── Mapping cat→num (si besoin)
CAT_TO_NUM = {
    "CODE_GENDER":      {"F": 0, "M": 1},
    "FLAG_OWN_REALTY":  {"N": 0, "Y": 1},
    "FLAG_OWN_CAR":     {"N": 0, "Y": 1},
}

for col, mp in CAT_TO_NUM.items():
    if col in df_full.columns:
        df_full[col] = df_full[col].map(mp)

# ────────────────────────── UI
st.title("Page 4 : Comparaison aux autres clients")
feature = st.selectbox("Variable à analyser :", VALID_FEATS, index=0)

client_val = USER_INPUT.get(feature)
if feature in CAT_TO_NUM:
    client_val = CAT_TO_NUM[feature].get(client_val, client_val)

vals   = df_full[feature]
shaps  = df_full[f"{feature}_shap"]

# ────────────────────────── Aide couleur
def color_from_shap(mean_val: float, eps: float = NEUTRAL_EPS) -> str:
    if mean_val >  eps: return "red"
    if mean_val < -eps: return "green"
    return "grey"

is_numeric = pd.api.types.is_numeric_dtype(vals) and vals.nunique() > 2

# ───────────── NUMÉRIQUE (histogramme) ─────────────
if is_numeric:
    # découpage manuel pour contrôler l’agrégation SHAP
    bin_edges  = np.histogram_bin_edges(vals.dropna(), bins="auto")
    bin_codes  = pd.cut(vals, bins=bin_edges, include_lowest=True)
    mean_shap  = shaps.groupby(bin_codes).mean()
    colors     = [color_from_shap(mean_shap.get(b, 0.0)) for b in bin_codes]

    fig = px.histogram(
        x=vals, nbins=len(bin_edges)-1, color_discrete_sequence=["lightgrey"],
        title=f"Distribution de **{feature}**"
    )
    fig.update_traces(marker_color=colors, selector=dict(type="histogram"))
    fig.add_vline(x=float(client_val), line_dash="dash", line_color="white",
                  annotation_text="Client", annotation_position="top right")
    fig.update_layout(xaxis_title=feature, yaxis_title="Nombre de clients")

# ───────────── CATÉGORIEL / BINAIRE (barres) ────────
else:
    mean_cat = shaps.groupby(vals).mean()
    df_bar   = vals.value_counts(dropna=False).reset_index()
    df_bar.columns = [feature, "count"]
    df_bar["bar_color"] = df_bar[feature].map(lambda x: color_from_shap(mean_cat.get(x, 0.0)))

    fig = px.bar(
        df_bar, x=feature, y="count", color="bar_color",
        color_discrete_map={"red": "red", "green": "green", "grey": "grey"},
        title=f"Répartition de **{feature}**"
    )
    fig.update_traces(showlegend=False)
    if client_val in df_bar[feature].tolist():
        idx = df_bar[df_bar[feature] == client_val].index[0]
        fig.add_vline(x=idx, line_dash="dash", line_color="white",
                      annotation_text="Client", annotation_position="top right")
    fig.update_layout(yaxis_title="Nombre de clients")

# ────────────────────────── Affichage
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Couleurs : **vert** =favorable, **rouge** =défavorable, **gris** =impact neutre "
    f"(seuil ±{NEUTRAL_EPS}). — Ligne pointillée : valeur du client."
)
