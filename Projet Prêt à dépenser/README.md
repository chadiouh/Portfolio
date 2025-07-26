# Credit Scoring avec MLOps (API & Streamlit)

##  Contexte
Projet réalisé pour l'entreprise "Prêt à dépenser", visant à prédire le risque d’insolvabilité de clients sollicitant un crédit à la consommation, dans un contexte de faible historique bancaire. L’enjeu : automatiser les décisions tout en assurant transparence, suivi et robustesse du modèle via une démarche complète MLOps.

##  Objectifs
- Construire un modèle de scoring basé sur des données socio-financières.
- Optimiser un score métier pénalisant plus fortement les faux négatifs (risque de perte).
- Déployer une API de prédiction dans le cloud.
- Suivre le modèle dans le temps (MLFlow, Data Drift, Evidently).
- Créer une interface utilisateur pour tester l’API.

##  Outils & technologies
`LightGBM`, `scikit-learn`, `MLFlow`, `FastAPI`, `Streamlit`, `evidently`, `pandas`, `GitHub Actions`, `Pytest`, `Docker`, `Render`.

##  Étapes clés
1. **Feature engineering** et gestion des variables manquantes.
2. Entraînement de plusieurs modèles (baseline → LightGBM) + GridSearchCV.
3. Optimisation du **seuil métier** basé sur un coût asymétrique FN > FP.
4. Analyse de la **feature importance globale et locale** (SHAP).
5. Déploiement d’une API sur Render (FastAPI + Docker).
6. Interface de test utilisateur avec Streamlit.
7. Tracking des expérimentations via MLFlow + analyse de **data drift** (Evidently).
8. Intégration continue via GitHub Actions & tests unitaires.

##  Compétences clés
- Modélisation et évaluation orientée métier (score personnalisé).
- Feature engineering avancé et interprétabilité (SHAP).
- Mise en place d’un pipeline **MLOps complet** : de l’entraînement au monitoring.
- Déploiement cloud, CI/CD, tests automatisés, gestion de projet data.

