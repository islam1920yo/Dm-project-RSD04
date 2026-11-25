"""
Module de prétraitement des données - Projet Data Mining 2026/2027
Auteurs : Équipe RSD04 (RSD04, CS12, CS22, CS29, IASD05, IASD25)
Ce module garantit l'absence totale de fuite d'information (data leakage)
en encapsulant toutes les transformations dans des pipelines Scikit-Learn.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Définition des colonnes du jeu de données Airline Passenger Satisfaction
NOMINAL_COLS = ['Gender', 'Customer Type', 'Type of Travel']
ORDINAL_COLS = ['Class']
CLASS_CATEGORIES = [['Economy', 'Economy Plus', 'Business']]

CONTINUOUS_COLS = ['Age', 'Flight Distance', 'Departure Delay', 'Arrival Delay']
SERVICE_RATING_COLS = [
    'Departure and Arrival Time Convenience', 'Ease of Online Booking',
    'Check-in Service', 'Online Boarding', 'Gate Location',
    'On-board Service', 'Seat Comfort', 'Leg Room Service', 'Cleanliness',
    'Food and Drink', 'In-flight Service', 'In-flight Wifi Service',
    'In-flight Entertainment', 'Baggage Handling'
]
NUMERIC_COLS = CONTINUOUS_COLS + SERVICE_RATING_COLS
TARGET_COL = 'Satisfaction'

def get_default_data_path():
    """Détecte l'emplacement du fichier CSV brut."""
    candidates = [
        "data/raw/airline_passenger_satisfaction.csv",
        "Dm-project-RSD04/data/raw/airline_passenger_satisfaction.csv",
        "airline_passenger_satisfaction.csv"
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Le fichier de données 'airline_passenger_satisfaction.csv' est introuvable.")

def load_raw_data(filepath=None):
    """Charge le jeu de données brut depuis le disque."""
    if filepath is None:
        filepath = get_default_data_path()
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """
    Nettoyage initial :
    - Suppression de la colonne ID (identifiant sans valeur informative prédictive)
    - Conservation de la cible pour l'étape de séparation
    """
    df_clean = df.copy()
    if 'ID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['ID'])
    return df_clean

def build_preprocessor():
    """
    Construit un ColumnTransformer scikit-learn complet :
    - Numériques (continues + notes) : Imputation par la médiane + StandardScaler (Z-score)
    - Ordinales (Class) : Imputation mode + OrdinalEncoder + StandardScaler
    - Nominales : Imputation mode + OneHotEncoder(drop='first')
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=CLASS_CATEGORIES)),
        ('scaler', StandardScaler())
    ])

    nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERIC_COLS),
            ('ord', ordinal_pipeline, ORDINAL_COLS),
            ('nom', nominal_pipeline, NOMINAL_COLS)
        ],
        remainder='drop'
    )
    return preprocessor

def prepare_data_splits(df, test_size=0.20, random_state=42):
    """
    Effectue un découpage train/test stratifié sans fuite de données,
    ajuste le préprocesseur sur le train set uniquement,
    et renvoie les matrices transformées ainsi que les cibles.
    """
    df_clean = clean_data(df)
    y_raw = df_clean[TARGET_COL]
    # Encodage binaire de la cible : Satisfied = 1, Neutral or Dissatisfied = 0
    y = (y_raw == 'Satisfied').astype(int)
    X = df_clean.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    return {
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'X_train_trans': X_train_trans,
        'X_test_trans': X_test_trans,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor
    }

def get_feature_names(preprocessor):
    """Renvoie les noms des caractéristiques après transformation."""
    nom_features = preprocessor.named_transformers_['nom'].named_steps['encoder'].get_feature_names_out(NOMINAL_COLS).tolist()
    return NUMERIC_COLS + ORDINAL_COLS + nom_features

def transform_single_input(input_dict, preprocessor):
    """
    Transforme une saisie unique (ex. formulaire Streamlit)
    en matrice numpy compatible avec les modèles entraînés.
    """
    df_single = pd.DataFrame([input_dict])
    return preprocessor.transform(df_single)
