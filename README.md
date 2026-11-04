# ✈️ Prédiction de la satisfaction des passagers aériens

Projet de **Data Mining / Machine Learning classique** (sans Deep Learning ni LLM comme modèle) visant à prédire si un passager est **satisfait** ou **neutre/insatisfait** à partir de son profil, des caractéristiques de son vol et de ses notes de services.

Le pipeline complet — EDA, nettoyage, prétraitement, entraînement, évaluation — aboutit à une application web interactive (**Streamlit**) permettant d'explorer les données et de tester le modèle en direct.

📄 Le rapport détaillé du mini-projet est disponible dans `docs/Rapport_MiniProject.pdf`.

## 📑 Sommaire

1. [Objectif du projet](#-objectif-du-projet)
2. [Structure du projet](#️-structure-du-projet)
3. [Dataset](#-dataset)
4. [Démarche Data Mining](#-démarche-data-mining)
5. [Application Streamlit](#️-application-streamlit-srcapppy)
6. [Stack technique](#️-stack-technique)
7. [Installation](#️-installation)
8. [Reproductibilité](#-reproductibilité)
9. [Rôles des membres et contributions](#-rôles-des-membres-et-contributions)
10. [Résultats clés et synthèse](#-résultats-clés-et-synthèse)
11. [Limites](#️-limites)
12. [Pistes d'amélioration](#-pistes-damélioration)

---

## 📌 Objectif du projet

**Contexte métier.** Dans un secteur aérien très concurrentiel, la satisfaction des passagers influence la fidélisation et la réputation d'une compagnie. Elle dépend de nombreux facteurs : confort, services à bord, enregistrement, ponctualité, classe de voyage…

**Problématique.** *Peut-on prédire la satisfaction d'un passager à partir de son profil et de ses notes de services ? Quels facteurs pèsent le plus ?*

À partir du dataset **Airline Passenger Satisfaction** (129 880 enquêtes), prédire la variable cible binaire :

- `Satisfaction = 1` → le passager est **satisfait**
- `Satisfaction = 0` → le passager est **neutre ou insatisfait**

en utilisant uniquement des algorithmes de **Machine Learning classique**.

**Objectifs scientifiques**
- Réaliser la chaîne complète de Data Mining (End-to-End).
- Comparer les classifieurs du cours : Zero-R, K-NN, Arbre de décision, Random Forest, Naïve Bayes, SVM.
- Optimiser les hyperparamètres par `GridSearchCV` / `RandomizedSearchCV` (validation croisée stratifiée, K ≥ 5).
- Valider les relations observées par des tests statistiques justifiés.

**Objectifs applicatifs**
- Dashboard EDA interactif, prédiction en direct avec probabilité, comparaison des modèles.

---

## 🗂️ Structure du projet

```
Dm-project-RSD04/
│
├── data/
│   ├── raw/
│   │   ├── airline_passenger_satisfaction.csv   # Dataset brut (129 880 lignes, 24 colonnes)
│   │   └── data_dictionary.csv                  # Dictionnaire des variables
│   └── processed/
│       ├── airline_clean.csv                    # Dataset nettoyé (notebook 01)
│       └── hypothesis_tests.csv                 # Résultats des tests statistiques (notebook 01)
│
├── docs/
│   └── Rapport_MiniProject.pdf                  # Rapport écrit du mini-projet
│
├── models/
│   ├── best_model.pkl                           # Modèle final (joblib)
│   └── preprocessor.pkl                         # ColumnTransformer (imputation, scaling, OHE)
│
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb                # Exploration, nettoyage, tests statistiques
│   ├── 02_preprocessing_encoding.ipynb          # Split train/test, pipeline de prétraitement
│   └── 03_model_training_evaluation.ipynb       # Entraînement, comparaison, sauvegarde du modèle
│
├── src/
│   ├── app.py                                   # Application Streamlit (point d'entrée)
│   ├── preprocessing.py                         # Fonctions de nettoyage et de préparation
│   └── models.py                                # Chargement des modèles et prédiction
│
├── .gitignore
├── README.md
└── requirements.txt
```

> 💡 Après avoir cloné le dépôt, exécute les notebooks dans l'ordre (01 → 02 → 03) pour régénérer `data/processed/` et les fichiers `models/*.pkl`. L'application résout les chemins à partir de la racine du projet (`Path(__file__).resolve().parents[1]`), aucune configuration manuelle n'est nécessaire une fois les modèles générés.

---

## 📊 Dataset

- **Source** : Airline Passenger Satisfaction (Kaggle, `mysarahmadbhat`) — https://www.kaggle.com/datasets/mysarahmadbhat/airline-passenger-satisfaction
- **Volume** : 129 880 observations brutes, 24 variables (≥ 10 000 lignes et ≥ 10 attributs : exigence du cahier des charges respectée)
- **Variable cible** : `Satisfaction` (classification binaire)
- **Équilibre des classes** : 56,55 % neutre/insatisfait (73 452) vs 43,45 % satisfait (56 428) : jeu de données modérément équilibré, baseline Zero-R ≈ 56,55 % d'accuracy
- **Valeurs manquantes** : uniquement `Arrival Delay` (393 lignes) ; aucun doublon
- **Stockage** : fichier < 100 Mo, hébergé directement dans `data/raw/`

| Colonne | Type | Description |
|---|---|---|
| ID | identifiant | Identifiant passager (supprimé) |
| Gender | catégorielle | Female / Male |
| Age | quantitative | Âge du passager |
| Customer Type | catégorielle | First-time / Returning |
| Type of Travel | catégorielle | Business / Personal |
| Class | catégorielle | Business / Economy / Economy Plus |
| Flight Distance | quantitative | Distance du vol (en miles) |
| Departure Delay | quantitative | Retard au départ (minutes) |
| Arrival Delay | quantitative | Retard à l'arrivée (minutes) |
| 14 notes de service | ordinale (1–5, 0 = non applicable) | Seat Comfort, Cleanliness, Online Boarding, In-flight Wifi Service, In-flight Entertainment, Check-in Service, Baggage Handling, Food and Drink, Leg Room Service, On-board Service, In-flight Service, Gate Location, Ease of Online Booking, Departure and Arrival Time Convenience |
| Satisfaction | cible | Satisfied / Neutral or Dissatisfied |

> Les notes de service vont de 1 (le plus bas) à 5 (le plus haut) ; **0 signifie « non applicable »** (`data_dictionary.csv`). Colonnes concernées : surtout `Departure and Arrival Time Convenience`, `Ease of Online Booking`, `Online Boarding`, `In-flight Wifi Service`, `Leg Room Service`.

---

## 🔬 Démarche Data Mining

### 1️⃣ EDA & Nettoyage (`01_eda_and_cleaning.ipynb`)

- Exploration structurelle du dataset, typage des variables (identifiant, quantitatives, ordinales, catégorielles, cible)
- Analyse des valeurs manquantes et des doublons
- Statistiques descriptives, distributions (histogrammes, KDE), détection des outliers (règle de Tukey / IQR)
- Analyse de la variable cible et de son équilibre (baseline Zero-R)
- Matrices de corrélation (Pearson et Spearman)
- **Tests statistiques (α = 0,05)** :

| Test | Variables | Justification du choix |
|---|---|---|
| H1 — Mann-Whitney U | `Age` vs `Satisfaction` | Quantitative vs binaire, non-normalité (Shapiro-Wilk) |
| H2 — Mann-Whitney U | `Flight Distance` vs `Satisfaction` | Variable asymétrique, deux groupes indépendants |
| H3 — Chi² d'indépendance | `Class` vs `Satisfaction` | Deux variables catégorielles |
| H4 — Chi² d'indépendance | `Type of Travel` vs `Satisfaction` | Deux variables catégorielles |
| H5 — Spearman | Deux notes de service | Variables ordinales, relation monotone |

  Chaque test fournit : statistique, p-value brute, **p-value ajustée (FDR Benjamini-Hochberg)**, taille d'effet (r de rang bisérial, V de Cramér, ρ) et décision. Avec ~130 000 lignes, la taille d'effet est déterminante pour l'interprétation. Les résultats sont repris dans l'onglet 🧪 *Tests d'hypothèses* de l'application.
- Nettoyage : suppression de l'identifiant, des doublons, espaces et chaînes vides (→ `Unknown`), suppression des lignes à cible manquante, encodage de la cible
- Sauvegarde → `data/processed/airline_clean.csv` et `data/processed/hypothesis_tests.csv`

### 2️⃣ Prétraitement & Encodage (`02_preprocessing_encoding.ipynb`)

- Séparation X / y, suppression de `ID`
- Split train/test **stratifié** (80 % / 20 %, `random_state=42`)
- Construction d'un pipeline `ColumnTransformer` combinant :
  - Numériques et notes de service → `SimpleImputer(median)` + `StandardScaler`
  - Catégorielles (Gender, Customer Type, Type of Travel, Class) → `SimpleImputer(most_frequent)` + `OneHotEncoder`
- `fit_transform` sur le train uniquement, `transform` sur le test (**aucune fuite de données**) ; l'imputation de `Arrival Delay` est faite ici, dans le pipeline
- Sauvegarde → `models/preprocessor.pkl`

### 3️⃣ Modélisation & Évaluation (`03_model_training_evaluation.ipynb`)

Algorithmes comparés, tous optimisés par `GridSearchCV` / `RandomizedSearchCV` (validation croisée stratifiée, K ≥ 5), Zero-R exempté :

| Modèle | Hyperparamètres étudiés |
|---|---|
| Zero-R (baseline) | — |
| K-NN | k, métrique de distance, pondération |
| Arbre de décision | critère (Gini / entropie), profondeur, élagage |
| Random Forest | nombre d'arbres, profondeur |
| Naïve Bayes | lissage (`var_smoothing` / `alpha`) |
| SVM | noyau (linéaire / RBF), C, γ |

> Le dataset comporte ~130 000 lignes : pour les modèles coûteux (SVM RBF, K-NN), le réglage peut s'appuyer sur `RandomizedSearchCV` ou sur un sous-échantillon stratifié. Ce choix est justifié dans le rapport.

Métriques : Accuracy, Precision, Recall, F1-Score, ROC-AUC, matrices de confusion, courbes ROC, importance des variables.
Le meilleur modèle est sérialisé (`joblib`) dans `models/best_model.pkl`.

---

## 🖥️ Application Streamlit (`src/app.py`)

Interface interactive avec plusieurs onglets :

- 🏠 **Accueil** — présentation du projet
- 📊 **Dashboard EDA** — statistiques descriptives, valeurs manquantes, distributions (histogrammes + KDE), fréquences catégorielles, matrice de corrélation, **filtres interactifs** (classe, type de voyage, tranche d'âge)
- 🔮 **Prédiction en Direct** — saisie d'un profil (âge, classe, type de voyage, distance, notes de services) → prédiction en temps réel avec probabilité, via le modèle appliqué aux données transformées par `preprocessor.pkl`
- 🏆 **Performance des Modèles** — tableau comparatif, courbes ROC, matrices de confusion
- 🧪 **Tests d'hypothèses** — tableau récapitulatif des tests du notebook 01 (α = 0,05) : statistique, p-value brute, p-value ajustée (FDR Benjamini-Hochberg), taille d'effet et décision
- 💡 **Découvertes Clés** — synthèse des insights (variables les plus influentes, modèle champion, apport du tuning)
- ℹ️ **À propos** — stack technique et avertissement académique

### Lancer l'application

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

L'application charge automatiquement `models/best_model.pkl` et `models/preprocessor.pkl` (mise en cache via `@st.cache_resource`).

---

## 🛠️ Stack technique

Python · pandas · NumPy · scikit-learn · SciPy (tests statistiques) · GridSearchCV / RandomizedSearchCV · joblib · Streamlit · matplotlib / seaborn (visualisation)

Aucune bibliothèque de Deep Learning (pas de TensorFlow / PyTorch / Keras) et aucun LLM comme modèle prédictif — uniquement du **Machine Learning classique**.

---

## ⚙️ Installation

```bash
git clone https://github.com/islam1920yo/Dm-project-RSD04.git
cd Dm-project-RSD04
python -m venv venv
source venv/bin/activate        # Windows (Git Bash) : source venv/Scripts/activate
pip install -r requirements.txt
```

**Accès aux données** : le fichier `airline_passenger_satisfaction.csv` (et `data_dictionary.csv`) est inclus dans `data/raw/`. À défaut, le télécharger depuis Kaggle (lien dans la section Dataset) et le placer dans `data/raw/`.

Exécuter ensuite les notebooks dans l'ordre 01 → 02 → 03, ou lancer directement `streamlit run src/app.py` si les fichiers `.pkl` sont déjà présents dans `models/`.

---

## 🔁 Reproductibilité

Pour régénérer entièrement le projet à partir des données brutes :

| Étape | Notebook | Entrée | Sortie |
|---|---|---|---|
| 1. Nettoyage | `01_eda_and_cleaning.ipynb` | `data/raw/airline_passenger_satisfaction.csv` | `data/processed/airline_clean.csv`, `hypothesis_tests.csv` |
| 2. Prétraitement | `02_preprocessing_encoding.ipynb` | `data/processed/airline_clean.csv` | matrices train/test + `models/preprocessor.pkl` |
| 3. Modélisation | `03_model_training_evaluation.ipynb` | matrices train/test | `models/best_model.pkl` |
| 4. Démo | `src/app.py` | `models/*.pkl` | interface Streamlit |

Un `random_state=42` est fixé sur les modèles à composante aléatoire et le split train/test est stratifié (`stratify=y`) pour garantir la reproductibilité des résultats.

---

## 👥 Rôles des membres et contributions

> Conformément aux règles du projet, seuls les **codes étudiants** sont utilisés (aucun nom réel).

| Code étudiant | Rôle | Responsabilités |
|---|---|---|
| RSD04 *(Team Leader)* | Coordination, rapport | Suivi du projet, emails officiels, relecture des Pull Requests, rapport PDF |
| CS12 | Prétraitement | Notebook 02 : pipeline de prétraitement (imputation, standardisation, encodage), split train/test |
| CS22 | Modélisation | Notebook 03 : Zero-R, K-NN, Arbre de décision, Random Forest, optimisation (GridSearchCV) |
| CS29 | Modélisation et évaluation | Notebook 03 : Naïve Bayes, SVM, métriques, courbes ROC, matrices de confusion |
| IASD05 | EDA, tests statistiques, documentation | Notebook 01 (EDA, nettoyage, tests statistiques), README, partie EDA du rapport |
| IASD25 | Application | Application Streamlit (`src/app.py`, `models.py`) : dashboard EDA et prédiction |

---

## 📈 Résultats clés et synthèse

### Comparaison des modèles *(à compléter avec les résultats du notebook 03)*

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Zero-R (Baseline) | | | | | N/A |
| K-Nearest Neighbors | | | | | |
| Decision Tree | | | | | |
| Random Forest | | | | | |
| Naïve Bayes | | | | | |
| SVM | | | | | |

**Modèle champion** : *à compléter* (sérialisé dans `models/best_model.pkl`).

### Justification des choix

- **Optimisation des hyperparamètres** : `GridSearchCV` / `RandomizedSearchCV` avec validation croisée stratifiée (K ≥ 5) pour tous les modèles réglables ; le baseline Zero-R est exempté. *(Détailler les grilles retenues et les meilleurs paramètres.)*
- **Tests statistiques** : tests non paramétriques (Mann-Whitney U, Spearman) car les variables ne suivent pas une loi normale (Shapiro-Wilk) ; Chi² pour les associations entre variables catégorielles ; correction FDR de Benjamini-Hochberg pour limiter les faux positifs lors de tests multiples ; tailles d'effet systématiquement reportées.
- **Métriques** : le baseline Zero-R donne la référence minimale ; F1-Score et ROC-AUC complètent l'accuracy.

### Résultats des tests statistiques (notebook 01, α = 0,05)

| Test | Variables | Statistique | p ajustée (FDR) | Taille d'effet | Décision |
|---|---|---|---|---|---|
| H1 — Mann-Whitney U | Age vs Satisfaction | 2,42·10⁹ | < 10⁻³⁰⁰ | \|r\| = 0,17 (petit) | Rejet de H0 |
| H2 — Mann-Whitney U | Flight Distance vs Satisfaction | 2,69·10⁹ | < 10⁻³⁰⁰ | \|r\| = 0,30 (petit) | Rejet de H0 |
| H3 — Chi² | Class vs Satisfaction | 32 906 | < 10⁻³⁰⁰ | V = 0,50 (grand) | Rejet de H0 |
| H4 — Chi² | Type of Travel vs Satisfaction | 26 283 | < 10⁻³⁰⁰ | V = 0,45 (moyen) | Rejet de H0 |
| H5 — Spearman | In-flight Wifi Service vs Online Boarding | ρ = 0,44 | < 10⁻³⁰⁰ | \|ρ\| = 0,44 (moyen) | Rejet de H0 |

Avec ~130 000 lignes, toutes les p-values sont extrêmement faibles : l'interprétation repose donc sur la **taille d'effet**. La classe de voyage (V = 0,50) et le type de voyage (V = 0,45) sont les associations les plus fortes avec la satisfaction.

### Découvertes clés *(à compléter après le notebook 03)*

- Variables les plus influentes sur la satisfaction : *…*

---

## ⚠️ Limites

- Les données proviennent d'enquêtes : les notes de services sont subjectives.
- La signification des notes à `0` (« non applicable ») demande un traitement soigneux.
- Dataset modérément déséquilibré (56,55 / 43,45) : surveiller le Recall et le F1-Score des deux classes.
- Application développée à des fins académiques ; les prédictions sont indicatives.

## 🚀 Pistes d'amélioration

- Étendre le benchmark à XGBoost, LightGBM ou Extra Trees (ML classique autorisé).
- Analyse d'explicabilité (permutation importance, SHAP) des facteurs de satisfaction.
- Ajout d'un assistant LLM *dans l'interface uniquement* (explication des prédictions), sans l'utiliser comme modèle prédictif.
- Déploiement en ligne de l'application (Streamlit Community Cloud).

---

## 👤 Contexte

Projet réalisé dans le cadre du module **Data Mining (Fouille de Données)** — Année académique 2026/2027.
