"""
Application Web Interactive de Data Mining - Satisfaction Passagers Aériens
Projet Année Académique 2026/2027
Auteurs : Équipe RSD04 (RSD04, CS12, CS22, CS29, IASD05, IASD25)
Classification Supervisée - Modèles ML Classiques Scikit-Learn
Interface UI/UX Stylisée Haute Fidélité
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Ajout du chemin pour l'import des modules locaux
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from preprocessing import load_raw_data, clean_data, NOMINAL_COLS, ORDINAL_COLS, NUMERIC_COLS
from models import (
    load_metrics_comparison, load_best_hyperparameters,
    load_statistical_tests, load_feature_importances,
    load_pipeline, predict_single, explain_prediction_heuristic, MODEL_FILES
)

# Configuration de la page Streamlit
st.set_page_config(
    page_title="SkyMining — Airline Customer Satisfaction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Feuille de style CSS moderne & Design System Aéronautique
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* En-tête Principal Stylisé */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 60%, #2563EB 100%);
        border-radius: 16px;
        padding: 30px 35px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #E2E8F0;
        font-weight: 400;
        max-width: 850px;
        line-height: 1.5;
        margin-bottom: 12px;
    }
    .hero-badges {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Cartes Métriques KPI */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin-bottom: 25px;
    }
    .kpi-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 5px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
    }
    .kpi-delta {
        font-size: 0.8rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 4px;
        margin-top: 4px;
    }
    .delta-positive { color: #10B981; }
    .delta-neutral { color: #64748B; }

    /* Cartes de sections */
    .section-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 22px 25px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        margin-bottom: 22px;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Boîtes de Résultat de Prédiction */
    .pred-card-satisfied {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 8px 16px -4px rgba(16, 185, 129, 0.2);
    }
    .pred-card-dissatisfied {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 8px 16px -4px rgba(239, 68, 68, 0.2);
    }
    .pred-status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 30px;
        font-weight: 800;
        font-size: 1.2rem;
        letter-spacing: 0.03em;
        margin-bottom: 10px;
    }
    .status-satisfied {
        background: #059669;
        color: #FFFFFF;
    }
    .status-dissatisfied {
        background: #DC2626;
        color: #FFFFFF;
    }

    /* Style de la barre latérale */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    .sidebar-brand {
        padding: 10px 0 20px 0;
        text-align: center;
    }
    .sidebar-brand h2 {
        font-size: 1.4rem;
        font-weight: 800;
        color: #1E3A8A;
        margin: 0;
    }
    .sidebar-brand p {
        font-size: 0.8rem;
        color: #64748B;
        margin: 0;
    }

    /* Amélioration des onglets Streamlit */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
        color: #64748B;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
        border-bottom: 2px solid #2563EB !important;
    }
</style>
""", unsafe_allow_html=True)

# Mise en cache des données
@st.cache_data
def get_cached_raw_data():
    return load_raw_data()

@st.cache_data
def get_cached_metrics():
    return load_metrics_comparison()

@st.cache_data
def get_cached_params():
    return load_best_hyperparameters()

@st.cache_data
def get_cached_stats():
    return load_statistical_tests()

@st.cache_data
def get_cached_feat_importances():
    return load_feature_importances()

@st.cache_resource
def get_cached_pipeline(model_name):
    return load_pipeline(model_name)

# Chargement initial des données
try:
    df_raw = get_cached_raw_data()
except Exception as e:
    st.error(f"Erreur de chargement des données : {e}")
    st.stop()

# Barre latérale professionnelle
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>✈️ SkyMining UI</h2>
        <p>Data Mining & Classification Supervisée</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Vue Générale & Métier",
            "📊 Dashboard EDA Interactif",
            "📐 Tests Statistiques (FDR)",
            "🤖 Benchmark des Modèles",
            "🔮 Cockpit de Prédiction"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="background:#F1F5F9; border-radius:10px; padding:12px; font-size:0.82rem; color:#475569; border:1px solid #E2E8F0;">
        <b>🎓 Projet Académique :</b> 2026/2027<br/>
        <b>👥 Équipe RSD04 :</b><br/>
        • <code>RSD04</code> (Team Leader)<br/>
        • <code>CS12</code>, <code>CS22</code>, <code>CS29</code><br/>
        • <code>IASD05</code>, <code>IASD25</code><br/>
        <b>🔒 Confidentialité :</b> Anonymat strict
    </div>
    """, unsafe_allow_html=True)

# Configuration graphique Matplotlib globale
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8

# ==========================================
# PAGE 1 : VUE GÉNÉRALE & MÉTIER
# ==========================================
if page == "🏠 Vue Générale & Métier":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">✈️ Prédiction de la Satisfaction des Passagers Aériens</div>
        <div class="hero-subtitle">
            Système décisionnel de Data Mining appliqué à la modélisation prédictive de l'expérience client dans le transport aérien.
            Déploiement complet sans fuite de données avec Scikit-Learn.
        </div>
        <div class="hero-badges">
            <span class="hero-badge">📦 129 880 Passagers Bruts</span>
            <span class="hero-badge">🎯 24 Attributs Multivariés</span>
            <span class="hero-badge">🏆 Champion : Random Forest (96.29%)</span>
            <span class="hero-badge">🛡️ Contrôle FDR Benjamini-Hochberg</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Cartes KPI Haut de Page
    n_total = len(df_raw)
    n_sat = (df_raw['Satisfaction'] == 'Satisfied').sum()
    pct_sat = (n_sat / n_total) * 100
    avg_dist = int(df_raw['Flight Distance'].mean())
    avg_delay = round(float(df_raw['Arrival Delay'].dropna().mean()), 1)

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Cohorte Analysée</div>
            <div class="kpi-value">{n_total:,}</div>
            <div class="kpi-delta delta-positive">✓ Conformité ≥ 10 000 lignes</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Taux de Satisfaction</div>
            <div class="kpi-value">{pct_sat:.1f}%</div>
            <div class="kpi-delta delta-neutral">{n_sat:,} passagers satisfaits</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Distance Moyenne Vol</div>
            <div class="kpi-value">{avg_dist:,} <span style="font-size:1rem; font-weight:500;">miles</span></div>
            <div class="kpi-delta delta-neutral">Court, Moyen & Long Courriers</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Retard Moyen Arrivée</div>
            <div class="kpi-value">{avg_delay} <span style="font-size:1rem; font-weight:500;">min</span></div>
            <div class="kpi-delta delta-neutral">Médiane = 0 min (Tukey Outliers)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">🎯 Enjeux Stratégiques & Méthodologie CRISP-DM</div>
            <p style="color:#475569; line-height:1.6; font-size:0.95rem;">
                Dans l'industrie aérienne, l'attrition client (<i>churn</i>) génère des pertes financières directes colossales. 
                Ce projet déploie une chaîne de valeur complète de <b>Data Mining</b> combinant :
            </p>
            <ul style="color:#334155; line-height:1.7; font-size:0.92rem;">
                <li><b>Prétraitement certifié sans fuite :</b> Imputation robuste par la médiane, encodage One-Hot/Ordinal et standardisation Z-score encapsulés dans un <code>ColumnTransformer</code> ajusté strictement sur le train set.</li>
                <li><b>Validation Inférentielle préalable :</b> 26 tests statistiques (D'Agostino-Pearson, Mann-Whitney U, Chi² + V de Cramér, Spearman) corrigés pour comparaisons multiples par le <b>FDR de Benjamini-Hochberg</b>.</li>
                <li><b>Benchmark des 5 familles du cours :</b> Zero-R, K-NN, Arbre de Décision, Random Forest, Naïve Bayes et SVM, tous optimisés sous validation croisée stratifiée 5 plis.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">📋 Répartition de la Cible (Équilibre)</div>
        """, unsafe_allow_html=True)
        fig_pie, ax_pie = plt.subplots(figsize=(4, 3))
        colors_pie = ['#3B82F6', '#F59E0B']
        ax_pie.pie(
            [n_total - n_sat, n_sat],
            labels=['Neutre / Insatisfait', 'Satisfait'],
            autopct='%1.1f%%',
            startangle=140,
            colors=colors_pie,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        ax_pie.axis('equal')
        st.pyplot(fig_pie)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🗂️ Échantillon des Données Brutes (24 Attributs)")
    st.dataframe(df_raw.head(8), use_container_width=True)

# ==========================================
# PAGE 2 : DASHBOARD EDA INTERACTIF
# ==========================================
elif page == "📊 Dashboard EDA Interactif":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📊 Dashboard d'Exploration Interactive (EDA)</div>
        <div class="hero-subtitle">
            Exploration visuelle et statistique conforme aux 6 exigences minimales obligatoires.
            Utilisez les filtres dynamiques latéraux pour segmenter instantanément les visualisations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Filtres Dynamiques (Exigence 6)
    st.sidebar.markdown("### 🔍 Filtres Dynamiques Actifs")
    class_filter = st.sidebar.multiselect(
        "Classe de voyage :",
        options=list(df_raw['Class'].unique()),
        default=list(df_raw['Class'].unique())
    )
    travel_type_filter = st.sidebar.multiselect(
        "Motif de déplacement :",
        options=list(df_raw['Type of Travel'].unique()),
        default=list(df_raw['Type of Travel'].unique())
    )
    age_range = st.sidebar.slider(
        "Tranche d'âge passager :",
        min_value=int(df_raw['Age'].min()),
        max_value=int(df_raw['Age'].max()),
        value=(18, 70)
    )

    # Application du filtrage
    filtered_df = df_raw[
        (df_raw['Class'].isin(class_filter)) &
        (df_raw['Type of Travel'].isin(travel_type_filter)) &
        (df_raw['Age'] >= age_range[0]) &
        (df_raw['Age'] <= age_range[1])
    ]

    st.markdown(f"""
    <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px; padding:12px 18px; margin-bottom:20px; color:#1E40AF; font-size:0.92rem; font-weight:600;">
        ⚡ <b>Filtre dynamique actif :</b> {len(filtered_df):,} passagers sélectionnés ({len(filtered_df)/len(df_raw)*100:.1f}% de la base globale).
    </div>
    """, unsafe_allow_html=True)

    tab_desc, tab_dist, tab_cat, tab_corr, tab_miss = st.tabs([
        "1️⃣ Statistiques Descriptives",
        "2️⃣ Distributions & KDE",
        "3️⃣ Fréquences Catégorielles",
        "4️⃣ Heatmap de Corrélation",
        "5️⃣ Données Manquantes"
    ])

    with tab_desc:
        st.markdown("#### 📐 Métriques Complètes de Tendance Centrale et Dispersion (Exigence 1)")
        st.dataframe(filtered_df.describe().T.style.background_gradient(cmap='Blues', subset=['mean', 'std']), use_container_width=True)

    with tab_dist:
        st.markdown("#### 📈 Histogrammes et Densités KDE des Variables Quantitatives (Exigence 3)")
        c_sel, _ = st.columns([2, 3])
        with c_sel:
            selected_var = st.selectbox("Sélectionnez la variable :", ['Flight Distance', 'Age', 'Departure Delay', 'Arrival Delay'])
        
        fig_kde, ax_kde = plt.subplots(figsize=(10, 4))
        sns.histplot(filtered_df[selected_var].dropna(), kde=True, color='#2563EB', ax=ax_kde, bins=35)
        ax_kde.set_title(f"Densité KDE et Distribution Empirique de {selected_var}", fontsize=12, fontweight='bold', color='#1E293B')
        ax_kde.set_xlabel(selected_var, fontsize=10)
        ax_kde.set_ylabel("Effectif Passagers", fontsize=10)
        st.pyplot(fig_kde)

    with tab_cat:
        st.markdown("#### 📊 Fréquences Catégorielles et Taux de Satisfaction (Exigence 4)")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_bar1, ax_bar1 = plt.subplots(figsize=(6, 4))
            sns.countplot(data=filtered_df, x='Class', hue='Satisfaction', palette=['#3B82F6', '#10B981'], ax=ax_bar1)
            ax_bar1.set_title("Satisfaction selon la Classe de Siège", fontweight='bold', color='#1E293B')
            st.pyplot(fig_bar1)
        with col_c2:
            fig_bar2, ax_bar2 = plt.subplots(figsize=(6, 4))
            sns.countplot(data=filtered_df, x='Type of Travel', hue='Satisfaction', palette=['#3B82F6', '#10B981'], ax=ax_bar2)
            ax_bar2.set_title("Satisfaction selon le Motif de Déplacement", fontweight='bold', color='#1E293B')
            st.pyplot(fig_bar2)

    with tab_corr:
        st.markdown("#### 🧬 Matrice de Corrélation Numérique de Pearson (Exigence 5)")
        corr_subset = ['Age', 'Flight Distance', 'Departure Delay', 'Arrival Delay',
                       'Online Boarding', 'In-flight Wifi Service', 'Seat Comfort', 'Cleanliness']
        fig_heat, ax_heat = plt.subplots(figsize=(10, 5))
        sns.heatmap(filtered_df[corr_subset].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax_heat)
        ax_heat.set_title("Corrélations Linéaires entre Descripteurs Clés", fontweight='bold', color='#1E293B')
        st.pyplot(fig_heat)

    with tab_miss:
        st.markdown("#### 🔍 Visualisation et Taux d'Absence des Attributs (Exigence 2)")
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            miss_counts = df_raw.isnull().sum()
            miss_pct = (miss_counts / len(df_raw)) * 100
            df_miss = pd.DataFrame({'Valeurs Manquantes': miss_counts, 'Taux (%)': miss_pct})
            st.dataframe(df_miss[df_miss['Valeurs Manquantes'] > 0])
        with col_m2:
            st.markdown("""
            <div style="background:#F8FAFC; border-radius:10px; padding:15px; border:1px solid #E2E8F0; font-size:0.9rem; color:#475569;">
                <b>Justification Méthodologique du Traitement :</b><br/>
                Seul l'attribut <code>Arrival Delay</code> présente des valeurs manquantes (393 occurrences, soit 0.30% des données).
                Étant donné l'asymétrie extrême des retards (médiane à 0 min, queue étalée jusqu'à 1 584 min), une imputation par la <b>médiane</b>
                est scientifiquement justifiée pour préserver l'intégrité de la distribution sans supprimer d'observations.
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# PAGE 3 : TESTS STATISTIQUES (FDR)
# ==========================================
elif page == "📐 Tests Statistiques (FDR)":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📐 Validation Statistique Inférentielle</div>
        <div class="hero-subtitle">
            26 tests statistiques scientifiquement justifiés avec formalisation des hypothèses H0/H1,
            calculs de p-values exactes, tailles d'effet et correction de Benjamini-Hochberg (FDR).
        </div>
    </div>
    """, unsafe_allow_html=True)

    stat_data = get_cached_stats()
    if stat_data:
        df_stat_view = pd.DataFrame(stat_data)
        
        c_filter, _ = st.columns([2, 2])
        with c_filter:
            sel_family = st.selectbox(
                "Filtrer par famille de test :",
                ["Toutes les familles"] + list(df_stat_view['test'].unique())
            )
        
        if sel_family != "Toutes les familles":
            filtered_stat = df_stat_view[df_stat_view['test'] == sel_family]
        else:
            filtered_stat = df_stat_view

        st.markdown(f"#### 📋 Synthèse des Tests ({len(filtered_stat)} tests affichés)")
        st.dataframe(
            filtered_stat[['test', 'variable', 'stat', 'p_raw', 'p_adj_BH', 'effect_size', 'significant_0.05']].style.format({
                'stat': '{:.3f}',
                'p_raw': '{:.2e}',
                'p_adj_BH': '{:.2e}'
            }),
            use_container_width=True
        )

        st.markdown("#### 🔬 Fiches Méthodologiques Détaillées")
        for _, row in filtered_stat.head(6).iterrows():
            with st.expander(f"📌 {row['test']} — {row['variable']}"):
                st.markdown(f"**Hypothèse Nulle (H0) :** {row['H0']}")
                st.markdown(f"**Hypothèse Alternative (H1) :** {row['H1']}")
                st.markdown(f"**Statistique :** `{row['stat']:.4f}` | **p-value brute :** `{row['p_raw']:.2e}` | **p-value ajustée FDR :** `{row['p_adj_BH']:.2e}`")
                st.markdown(f"**Taille d'effet mesurée :** `{row['effect_size']}`")
                st.info(f"**Interprétation :** {row['interpretation']}")

# ==========================================
# PAGE 4 : BENCHMARK DES MODÈLES
# ==========================================
elif page == "🤖 Benchmark des Modèles":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🤖 Benchmark Multicritère des Algorithmes</div>
        <div class="hero-subtitle">
            Évaluation comparative rigoureuse des 5 familles d'algorithmes du cours + Baseline
            sur le jeu de test indépendant (25 976 passagers).
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_metrics = get_cached_metrics()
    best_params = get_cached_params()

    if df_metrics is not None:
        st.markdown("#### 🏆 Tableau Comparatif Multicritère")
        st.dataframe(
            df_metrics.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], color='#D1FAE5')
                             .highlight_min(subset=['Temps Fit (s)', 'Temps Inférence (s)'], color='#EFF6FF')
                             .format({
                                 'Accuracy': '{:.4f}',
                                 'Precision': '{:.4f}',
                                 'Recall': '{:.4f}',
                                 'F1-Score': '{:.4f}',
                                 'ROC-AUC': '{:.4f}',
                                 'Temps Fit (s)': '{:.2f}',
                                 'Temps Inférence (s)': '{:.3f}'
                             }),
            use_container_width=True
        )

        st.markdown("---")
        t_roc, t_cm, t_fi = st.tabs(["Courbes ROC Comparées", "Matrices de Confusion", "Importance des Variables"])
        
        with t_roc:
            roc_path = "data/figures/roc_curves_comparison.png"
            if not os.path.exists(roc_path):
                roc_path = "Dm-project-RSD04/data/figures/roc_curves_comparison.png"
            if os.path.exists(roc_path):
                st.image(roc_path, caption="Figure 1 : Discrimination maximale Random Forest (AUC = 0.9942)", use_container_width=True)

        with t_cm:
            cm_path = "data/figures/confusion_matrices_grid.png"
            if not os.path.exists(cm_path):
                cm_path = "Dm-project-RSD04/data/figures/confusion_matrices_grid.png"
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Figure 2 : Grille des matrices de confusion sur 25 976 données de test", use_container_width=True)

        with t_fi:
            fi_path = "data/figures/feature_importance_comparison.png"
            if not os.path.exists(fi_path):
                fi_path = "Dm-project-RSD04/data/figures/feature_importance_comparison.png"
            if os.path.exists(fi_path):
                st.image(fi_path, caption="Figure 3 : Importance triangulée (Gini, Permutation, Information Mutuelle)", use_container_width=True)

        st.markdown("#### ⚙️ Hyperparamètres Optimaux Identifiés par Validation Croisée 5 Plis")
        if best_params:
            c1, c2 = st.columns(2)
            for idx, (m_name, p) in enumerate(best_params.items()):
                target_col = c1 if idx % 2 == 0 else c2
                with target_col:
                    with st.expander(f"🔹 {m_name}"):
                        st.json(p)

# ==========================================
# PAGE 5 : COCKPIT DE PRÉDICTION
# ==========================================
elif page == "🔮 Cockpit de Prédiction":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🔮 Cockpit Prédictif & Stratification Temps Réel</div>
        <div class="hero-subtitle">
            Saisissez les paramètres de vol et d'évaluation d'un passager pour obtenir un diagnostic instantané
            avec jauge continue de confiance et explications opérationnelles.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_m, _ = st.columns([2, 2])
    with c_m:
        chosen_model = st.selectbox(
            "Algorithme actif :",
            list(MODEL_FILES.keys()),
            index=3 # Random Forest par défaut
        )

    pipeline = get_cached_pipeline(chosen_model)

    with st.form("cockpit_form"):
        st.markdown("### 1. Profil du Passager & Conditions de Vol")
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Genre :", ["Female", "Male"])
            customer_type = st.selectbox("Fidélité Client :", ["Returning", "First-time"])
            age = st.slider("Âge :", 7, 85, 41)
        with col2:
            travel_type = st.selectbox("Motif Déplacement :", ["Business", "Personal"])
            travel_class = st.selectbox("Classe :", ["Business", "Economy", "Economy Plus"])
            flight_distance = st.number_input("Distance de vol (miles) :", 50, 5000, 1050)
        with col3:
            dep_delay = st.number_input("Retard au départ (min) :", 0, 1500, 0)
            arr_delay = st.number_input("Retard à l'arrivée (min) :", 0, 1500, 0)
            time_conv = st.slider("Commodité des horaires (1-5) :", 1, 5, 4)

        st.markdown("### 2. Évaluation des Services Clés (Échelle 1 à 5)")
        c_s1, c_s2, c_s3 = st.columns(3)
        with c_s1:
            online_board = st.slider("Embarquement en Ligne (Online Boarding) :", 1, 5, 4)
            wifi_service = st.slider("Wi-Fi en Vol :", 1, 5, 4)
            online_book = st.slider("Réservation en Ligne :", 1, 5, 3)
            seat_comfort = st.slider("Confort du Siège :", 1, 5, 4)
        with c_s2:
            leg_room = st.slider("Espace aux Jambes (Leg Room) :", 1, 5, 4)
            entertainment = st.slider("Divertissement Multimédia :", 1, 5, 4)
            food_drink = st.slider("Restauration & Boissons :", 1, 5, 3)
            cleanliness = st.slider("Propreté Générale :", 1, 5, 4)
        with c_s3:
            onboard_serv = st.slider("Service du Personnel Navigant :", 1, 5, 4)
            checkin_serv = st.slider("Service d'Enregistrement :", 1, 5, 4)
            inflight_serv = st.slider("Service Global en Vol :", 1, 5, 4)
            baggage_hand = st.slider("Gestion des Bagages :", 1, 5, 4)
            gate_loc = st.slider("Localisation Porte :", 1, 5, 3)

        submit = st.form_submit_button("⚡ Calculer la Prédiction Instantanée", use_container_width=True)

    if submit:
        input_data = {
            'Gender': gender,
            'Customer Type': customer_type,
            'Age': age,
            'Type of Travel': travel_type,
            'Class': travel_class,
            'Flight Distance': flight_distance,
            'Departure Delay': dep_delay,
            'Arrival Delay': arr_delay,
            'Departure and Arrival Time Convenience': time_conv,
            'Ease of Online Booking': online_book,
            'Check-in Service': checkin_serv,
            'Online Boarding': online_board,
            'Gate Location': gate_loc,
            'On-board Service': onboard_serv,
            'Seat Comfort': seat_comfort,
            'Leg Room Service': leg_room,
            'Cleanliness': cleanliness,
            'Food and Drink': food_drink,
            'In-flight Service': inflight_serv,
            'In-flight Wifi Service': wifi_service,
            'In-flight Entertainment': entertainment,
            'Baggage Handling': baggage_hand
        }

        res = predict_single(pipeline, input_data)
        explanation = explain_prediction_heuristic(input_data, res)

        st.markdown("---")
        st.markdown("### 🎯 Diagnostic Prédictif")
        
        c_res1, c_res2 = st.columns([1, 2])
        with c_res1:
            if res['prediction_code'] == 1:
                st.markdown(f"""
                <div class="pred-card-satisfied">
                    <span class="pred-status-badge status-satisfied">PASSAGER SATISFAIT</span>
                    <h2 style="color:#065F46; margin:8px 0;">Confiance : {res['confidence']:.1f}%</h2>
                    <p style="color:#047857; font-size:0.95rem;">Probabilité : <b>{res['probability_satisfied']:.3f}</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-card-dissatisfied">
                    <span class="pred-status-badge status-dissatisfied">NEUTRE OU INSATISFAIT</span>
                    <h2 style="color:#991B1B; margin:8px 0;">Confiance : {res['confidence']:.1f}%</h2>
                    <p style="color:#B91C1C; font-size:0.95rem;">Probabilité : <b>{res['probability_dissatisfied']:.3f}</b></p>
                </div>
                """, unsafe_allow_html=True)
            
            st.write("")
            st.markdown(f"**Jauge Continue de Satisfaction :** `{res['probability_satisfied']*100:.1f}%`")
            st.progress(float(res['probability_satisfied']))

        with c_res2:
            st.markdown("""
            <div style="background:#FFFFFF; border-radius:14px; padding:20px; border:1px solid #E2E8F0; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05);">
                <div style="font-size:1.1rem; font-weight:700; color:#1E293B; margin-bottom:10px;">💬 Explication Décisionnelle & Recommandations Métier</div>
            """, unsafe_allow_html=True)
            st.info(explanation)
            st.markdown("</div>", unsafe_allow_html=True)
