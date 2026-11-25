# ✈️ Prédiction de la Satisfaction des Passagers Aériens par Data Mining

[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-red.svg)](https://streamlit.io/)
[![Licence](https://img.shields.io/badge/Licence-Académique-green.svg)]()

> **Projet de Fouille de Données (Data Mining) — Année Académique 2026/2027**  
> **Groupe :** `RSD04`  
> **Parcours Technique :** Parcours A — Classification Supervisée  
> **Règle d'Anonymisation :** Strict respect de la confidentialité, identification exclusive par codes étudiants (`RSD04`, `CS12`, `CS22`, `CS29`, `IASD05`, `IASD25`).

---

## SECTION 1 : Présentation, Objectifs et Description du Projet

### 1.1 Contexte Métier
Dans l'industrie du transport aérien, la satisfaction client est le moteur économique central déterminant la fidélisation des passagers, la valeur vie client (*Customer Lifetime Value*) et la réputation de marque. Face à une concurrence internationale féroce, les compagnies aériennes doivent identifier avec précision les facteurs critiques générant de l'insatisfaction (retards de vol, mauvaise ergonomie des plateformes numériques, inconfort à bord) afin de mener des actions préventives et ciblées.

### 1.2 Objectifs Scientifiques et Applicatifs
- **Objectifs Scientifiques :**
  1. Établir une méthodologie rigoureuse de Data Mining (KDD / CRISP-DM) exempte de toute fuite d'information (*Data Leakage Prevention*).
  2. Valider scientifiquement les hypothèses métier via des tests statistiques inférentiels adaptés et robustes, assortis d'une correction de taux de fausses découvertes (*Benjamini-Hochberg FDR*).
  3. Mettre en œuvre et comparer systématiquement les 5 grandes familles d'algorithmes de classification du cours (plus une baseline), optimisées sous validation croisée stratifiée à 5 plis ($K \ge 5$).
- **Objectifs Applicatifs :**
  1. Développer une application décisionnelle web interactive sous **Streamlit** intégrant un dashboard EDA complet à 6 éléments minimaux.
  2. Offrir un module prédictif instantané permettant aux équipes de relation client d'évaluer la satisfaction d'un passager avec score de confiance et explication textuelle décisionnelle.

### 1.3 Présentation du Dataset
- **Source :** Jeu de données public *Airline Passenger Satisfaction* (Kaggle, par *mysarahmadbhat*).
- **Volumétrie Brute Initiale :**
  - **Nombre d'enregistrements bruts :** **129 880 lignes** (exigence $\ge 10\,000$ validée).
  - **Nombre d'attributs bruts :** **24 attributs** (exigence $\ge 10$ validée).
- **Variable Cible (`Satisfaction`) :** Binaire
  - `Neutral or Dissatisfied` : 73 452 passagers (56.55%)
  - `Satisfied` : 56 428 passagers (43.45%)
- **Attributs clés :**
  - Profil passager : `Gender`, `Customer Type`, `Age`, `Type of Travel`, `Class`.
  - Paramètres de vol : `Flight Distance`, `Departure Delay`, `Arrival Delay`.
  - Évaluations de services (1 à 5) : 14 dimensions incluant `In-flight Wifi Service`, `Online Boarding`, `Seat Comfort`, `In-flight Entertainment`, `Leg Room Service`, `Cleanliness`, etc.

### 1.4 Décisions et Choix Techniques Documentés
Conformément aux instructions, tout point sujet à arbitrage a été tranché selon l'option la plus conforme au cahier des charges :
1. **Valeurs manquantes sur `Arrival Delay` (393 lignes, 0.30%) :** En raison de la forte asymétrie positive des retards, l'imputation par la **médiane** au sein du pipeline scikit-learn a été retenue (robuste aux valeurs extrêmes, conservation de l'intégralité des observations d'entraînement).
2. **Encodage ordinal vs One-Hot :** L'attribut `Class` possède une hiérarchie intrinsèque (`Economy` < `Economy Plus` < `Business`) et est traité par `OrdinalEncoder` suivi d'une standardisation. Les variables nominales sans ordre (`Gender`, `Customer Type`, `Type of Travel`) sont traitées par `OneHotEncoder(drop='first')`.
3. **Complexité du SVM ($N \approx 130\,000$) :** L'entraînement d'un SVM à noyau RBF sur $130\,000$ exemples présente une complexité computationnelle en $\mathcal{O}(N^2)$ à $\mathcal{O}(N^3)$. Conformément au cahier des charges, le réglage d'hyperparamètres a été mené via `RandomizedSearchCV` sur un sous-échantillon stratifié représentatif (12 000 points), puis réentraîné sur un échantillon étendu (25 000 points), garantissant un compromis optimal entre puissance statistique et temps de calcul.
4. **Interdiction stricte du Deep Learning & LLM en tant que modèles :** Seuls des modèles de Machine Learning classiques scikit-learn sont utilisés. L'explication en langage naturel repose sur un moteur heuristique basé sur l'importance des variables, sans dépendance requise à une API payante.

---

## SECTION 2 : Guide d'Installation, Exécution Locale et Accès aux Données

### 2.1 Prérequis Système
- Python 3.12 (ou $\ge 3.10$)
- Gestionnaire de paquets `pip`
- Git

### 2.2 Procédure d'Installation Pas à Pas

1. **Cloner le dépôt GitHub :**
   ```bash
   git clone https://github.com/votre-compte/Dm-project-RSD04.git
   cd Dm-project-RSD04
   ```

2. **Créer et activer un environnement virtuel (recommandé) :**
   - *Sur Windows (PowerShell) :*
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - *Sur Linux / macOS :*
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Installer les dépendances exactes :**
   ```bash
   pip install -r requirements.txt
   ```

4. **Accès et Vérification des Données :**
   Les données brutes sont directement incluses dans le dépôt :
   - `data/raw/airline_passenger_satisfaction.csv` (12.8 Mo)
   - `data/raw/data_dictionary.csv`
   *(La taille étant inférieure au seuil de 100 Mo de GitHub, aucun téléchargement externe complexe n'est requis).*

5. **Exécuter l'application Streamlit :**
   ```bash
   streamlit run src/app.py
   ```
   L'application s'ouvre automatiquement dans votre navigateur par défaut à l'adresse locale `http://localhost:8501`.

6. **Déploiement en Ligne (Optionnel) :**
   L'application est prête pour un déploiement en 1 clic sur [Streamlit Community Cloud](https://streamlit.io/cloud) en pointant sur le fichier `src/app.py`.

---

## SECTION 3 : Rôles des Membres et Matrice des Contributions

Conformément à la règle de confidentialité stricte, seuls les **codes étudiants** officiels sont mentionnés. Le travail a été réparti de manière équitable et collaborative :

| Code Étudiant | Rôle & Responsabilité Principale | Modules & Livrables Pris en Charge | Taux d'Implication |
|---|---|---|:---:|
| **RSD04** *(Leader)* | Coordination globale, Architecture pipeline & Intégration | `src/preprocessing.py`, Validation croisée, Structure globale | 16.7% |
| **CS12** | Analyse Exploratoire des Données (EDA) & Visualisations | `notebooks/01_eda_and_cleaning.ipynb`, Graphiques univariés/bivariés | 16.7% |
| **CS22** | Nettoyage, Imputation & Encodage sans fuite de données | `notebooks/02_preprocessing_encoding.ipynb`, `data/processed/` | 16.7% |
| **CS29** | Tests Statistiques Inférentiels & Correction FDR | Formalisation $H_0/H_1$, D'Agostino, Mann-Whitney, Chi², Spearman, FDR | 16.7% |
| **IASD05** | Entraînement des Modèles, Hyper-tuning & Évaluation | `notebooks/03_model_training_evaluation.ipynb`, `src/models.py` | 16.7% |
| **IASD25** | Développement de l'Interface Utilisateur & Rapport PDF | `src/app.py` (Streamlit), `docs/Rapport_MiniProject.pdf`, `README.md` | 16.7% |

---

## SECTION 4 : Résultats Clés et Synthèse des Découvertes

### 4.1 Tableau Synthétique Comparatif des Modèles
Toutes les métriques ont été mesurées rigoureusement sur le jeu de test indépendant (**25 976 enregistrements**, split stratifié 80/20 avec `random_state=42`) :

| Famille d'Algorithme | Modèle Spécifique | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Temps Fit (s) | Temps Inf. (s) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | Zero-R (`DummyClassifier`) | 56.55% | 0.00% | 0.00% | 0.00% | 0.5000 | < 0.01 s | 0.001 s |
| **Instance-Based** | K-Nearest Neighbors (K-NN) | 93.48% | 94.45% | 90.30% | 92.33% | 0.9802 | 0.01 s | 7.935 s |
| **Arbres / Règles** | Decision Tree (Élagué) | 95.58% | 95.85% | 93.90% | 94.87% | 0.9878 | 85.84 s | 0.007 s |
| **Ensemble (Bagging)** | **Random Forest** 🏆 | **96.29%** | **97.26%** | **94.11%** | **95.66%** | **0.9942** | 139.57 s | 0.116 s |
| **Probabiliste** | Naive Bayes (`GaussianNB`) | 86.74% | 86.35% | 82.54% | 84.40% | 0.9253 | 2.38 s | 0.018 s |
| **Frontière de Marge** | Support Vector Machine (RBF) | 95.21% | 95.79% | 93.07% | 94.41% | 0.9877 | 50.11 s | 11.062 s |

### 4.2 Analyse des Résultats et Choix d'Optimisation
1. **Modèle Champion :** Le **Random Forest** surpasse l'ensemble des compétiteurs avec **96.29% d'exactitude** et un **ROC-AUC de 0.9942**. Les hyperparamètres optimaux retenus par validation croisée sont : `n_estimators=100`, `max_depth=25`, `min_samples_leaf=2`, critère `gini`.
2. **Performances du SVM :** Le SVM à noyau RBF optimisé atteint une excellente performance (**95.21% d'Accuracy, 0.9877 ROC-AUC** avec $C=5.0$, $\gamma=\text{'scale'}$), mais présente un coût d'inférence sensiblement plus élevé (~11 s sur le test set contre 0.12 s pour Random Forest).
3. **Importance Multi-Perspective des Variables :** Le croisement entre l'indice de Gini, la *Permutation Importance* et l'Information Mutuelle montre que les variables les plus déterminantes pour la satisfaction sont :
   - `Online Boarding` (Embarquement en ligne)
   - `In-flight Wifi Service` (Wi-Fi en vol)
   - `Type of Travel_Personal` (Voyage d'affaires vs personnel)
   - `Class_Business` (Voyage en classe Affaires)
   - `In-flight Entertainment` (Divertissement à bord)

### 4.3 Synthèse des Tests Statistiques
26 tests statistiques unitaires ont été formalisés et corrigés via Benjamini-Hochberg (FDR) :
- **Test de normalité (D'Agostino-Pearson) :** Rejet franc de la normalité sur toutes les variables quantitatives ($p < 10^{-100}$), justifiant rigoureusement le recours aux tests non paramétriques.
- **Mann-Whitney U :** Différences hautement significatives sur l'âge ($p < 10^{-50}$) et la distance de vol ($p < 10^{-100}$), les passagers satisfaits effectuant en moyenne des vols plus longs.
- **Test du Chi² + V de Cramér :** Association très forte entre `Type of Travel` et `Satisfaction` ($V = 0.45$, $p < 10^{-100}$) et entre `Class` et `Satisfaction` ($V = 0.49$, $p < 10^{-100}$).
- **Corrélation de Spearman :** Corrélations positives fortes entre les notes de service numériques (`Online Boarding`: $\rho = 0.50$, `In-flight Entertainment`: $\rho = 0.40$) et la satisfaction.
- **Contrôle FDR :** 100% des tests demeurent significatifs après correction de Benjamini-Hochberg au seuil $\alpha = 0.05$.
