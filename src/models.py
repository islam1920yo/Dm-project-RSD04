"""
Module de gestion et d'inférence des modèles ML - Projet Data Mining 2026/2027
Auteurs : Équipe RSD04 (RSD04, CS12, CS22, CS29, IASD05, IASD25)
Fournit le chargement des modèles, la prédiction avec probabilité/confiance
et les explications décisionnelles en langage naturel.
"""

import os
import json
import pandas as pd
import numpy as np
import joblib

MODEL_FILES = {
    'Zero-R (Baseline)': 'zero_r_baseline_pipeline.joblib',
    'K-NN': 'k_nn_pipeline.joblib',
    'Decision Tree': 'decision_tree_pipeline.joblib',
    'Random Forest': 'random_forest_pipeline.joblib',
    'Naive Bayes': 'naive_bayes_pipeline.joblib',
    'Support Vector Machine': 'support_vector_machine_pipeline.joblib'
}

def find_file(subpath):
    """Localise un fichier dans le dossier courant ou dans Dm-project-RSD04/."""
    candidates = [
        subpath,
        os.path.join("Dm-project-RSD04", subpath),
        os.path.join("..", subpath)
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return subpath

def load_metrics_comparison():
    """Charge le tableau comparatif des métriques."""
    path = find_file("data/processed/models_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_best_hyperparameters():
    """Charge le dictionnaire des hyperparamètres optimaux issus du tuning CV."""
    path = find_file("data/processed/best_hyperparameters.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_statistical_tests():
    """Charge les résultats des tests statistiques corrigés par Benjamini-Hochberg."""
    path = find_file("data/processed/statistical_tests.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def load_feature_importances():
    """Charge le tableau multi-perspective des importances des caractéristiques."""
    path = find_file("data/processed/feature_importances.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_pipeline(model_name):
    """Charge un pipeline complet (préprocesseur + modèle) par nom d'algorithme."""
    if model_name not in MODEL_FILES:
        raise ValueError(f"Modèle inconnu : {model_name}. Choix possibles : {list(MODEL_FILES.keys())}")
    
    filename = MODEL_FILES[model_name]
    path = find_file(os.path.join("models", filename))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier de pipeline introuvable : {path}. Veuillez exécuter l'entraînement préalablement.")
    
    return joblib.load(path)

def predict_single(pipeline, input_dict):
    """
    Effectue une prédiction sur une saisie passager unique.
    Renvoie un dictionnaire avec :
    - prediction_label ('Satisfied' ou 'Neutral or Dissatisfied')
    - prediction_code (1 ou 0)
    - probability_satisfied (flottant entre 0 et 1)
    - probability_dissatisfied (flottant entre 0 et 1)
    - confidence (pourcentage 0-100%)
    """
    df_single = pd.DataFrame([input_dict])
    pred_code = int(pipeline.predict(df_single)[0])
    
    proba_satisfied = 0.5
    proba_dissatisfied = 0.5
    
    clf = pipeline.named_steps['classifier']
    if hasattr(pipeline, 'predict_proba'):
        probas = pipeline.predict_proba(df_single)[0]
        # Ordre des classes : [0, 1]
        proba_dissatisfied = float(probas[0])
        proba_satisfied = float(probas[1])
    elif hasattr(clf, 'decision_function'):
        score = float(pipeline.decision_function(df_single)[0])
        # Sigmoïde pour projection en probabilité approchée
        proba_satisfied = 1.0 / (1.0 + np.exp(-score))
        proba_dissatisfied = 1.0 - proba_satisfied
    else:
        # Cas Zero-R ou classifieur sans proba
        if pred_code == 1:
            proba_satisfied = 0.85
            proba_dissatisfied = 0.15
        else:
            proba_satisfied = 0.15
            proba_dissatisfied = 0.85
            
    confidence = max(proba_satisfied, proba_dissatisfied) * 100.0
    label = "Satisfied" if pred_code == 1 else "Neutral or Dissatisfied"
    
    return {
        'prediction_label': label,
        'prediction_code': pred_code,
        'probability_satisfied': proba_satisfied,
        'probability_dissatisfied': proba_dissatisfied,
        'confidence': confidence
    }

def explain_prediction_heuristic(input_dict, prediction_result):
    """
    Explication ergonomique en langage naturel basée sur les règles métier 
    et les variables les plus influentes (Online Boarding, In-flight Wifi, Type of Travel, Class).
    Sert de moteur d'explication robuste et autonome sans dépendance à une clé API externe.
    """
    is_satisfied = prediction_result['prediction_code'] == 1
    conf = prediction_result['confidence']
    
    wifi = input_dict.get('In-flight Wifi Service', 3)
    boarding = input_dict.get('Online Boarding', 3)
    comfort = input_dict.get('Seat Comfort', 3)
    travel_type = input_dict.get('Type of Travel', 'Business')
    travel_class = input_dict.get('Class', 'Business')
    arr_delay = input_dict.get('Arrival Delay', 0)
    
    factors_positive = []
    factors_negative = []
    
    if boarding >= 4:
        factors_positive.append(f"Excellente expérience d'embarquement en ligne (note {boarding}/5)")
    elif boarding <= 2:
        factors_negative.append(f"Embarquement en ligne décevant (note {boarding}/5)")
        
    if wifi >= 4:
        factors_positive.append(f"Service Wi-Fi à bord très apprécié ({wifi}/5)")
    elif wifi <= 2:
        factors_negative.append(f"Wi-Fi à bord jugé insuffisant ({wifi}/5)")
        
    if travel_class == 'Business':
        factors_positive.append("Voyage en classe Affaires (confort et prestations supérieures)")
    elif travel_class == 'Economy':
        factors_negative.append("Voyage en classe Économique")
        
    if travel_type == 'Business':
        factors_positive.append("Déplacement professionnel (sensibilité accrue aux gains de temps)")
    else:
        factors_negative.append("Voyage personnel (sensibilité forte aux tarifs et retards)")
        
    if arr_delay > 30:
        factors_negative.append(f"Retard conséquent à l'arrivée ({arr_delay} minutes)")
    elif arr_delay == 0:
        factors_positive.append("Ponctualité parfaite à l'arrivée (0 min de retard)")

    lines = []
    if is_satisfied:
        lines.append(f"**Diagnostic : Passager globalement SATISFAIT (Confiance : {conf:.1f}%)**")
        lines.append("\n**Facteurs clés favorables ayant motivé cette prédiction :**")
        for f in factors_positive:
            lines.append(f"- ✅ {f}")
        if factors_negative:
            lines.append("\n*Points d'attention identifiés pouvant être améliorés :*")
            for f in factors_negative:
                lines.append(f"- ⚠️ {f}")
    else:
        lines.append(f"**Diagnostic : Passager NEUTRE OU INSATISFAIT (Confiance : {conf:.1f}%)**")
        lines.append("\n**Facteurs déterminants ayant causé l'insatisfaction :**")
        for f in factors_negative:
            lines.append(f"- ❌ {f}")
        if factors_positive:
            lines.append("\n*Aspects positifs relevés malgré l'insatisfaction globale :*")
            for f in factors_positive:
                lines.append(f"- ℹ️ {f}")
                
    lines.append("\n**Recommandation d'action pour la compagnie aérienne :**")
    if is_satisfied:
        lines.append("Fidéliser ce passager en maintenant la qualité des services numériques et du personnel navigant.")
    else:
        lines.append("Prioriser la fluidification du portail d'embarquement en ligne et la fiabilité du Wi-Fi en vol, leviers statistiques majeurs d'insatisfaction.")
        
    return "\n".join(lines)
