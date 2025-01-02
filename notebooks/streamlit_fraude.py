import joblib
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc
)

# === Configuration de la page ===
st.set_page_config(page_title="Détection de Fraude Bancaire", layout="wide", page_icon="💳")

# === Chargement des données ===
@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

df_creditcard = load_data()

# === Thème de l'application ===
def custom_css():
    st.markdown(
        """
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

custom_css()

# === Titre de l'application ===
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Détection de Fraude Bancaire</h1>", unsafe_allow_html=True)

# === Barre latérale ===
st.sidebar.title("Sommaire")
pages = ["Contexte du projet", "Exploration des données", "Analyse de données", "Modélisation"]
page = st.sidebar.radio("Aller vers la page :", pages)

# === Contexte du projet ===
if page == pages[0]: 
    # Contexte du projet
    st.write("### Contexte du projet")
    st.write("""
    Ce projet s'inscrit dans un contexte bancaire. L'objectif est de développer une solution 
    de détection de fraude bancaire en utilisant l'analyse de données et le machine learning avec Python.
    """)
    st.write("""
    Nous avons à notre disposition le fichier `creditcard.csv` contenant des données de transactions bancaires. 
    Chaque observation en ligne correspond à une transaction, et chaque variable en colonne est une caractéristique de la transaction.
    """)
    st.write("""
    Dans un premier temps, nous explorerons ce dataset. Ensuite, nous analyserons visuellement les données, 
    avant d'implémenter des modèles de Machine Learning pour prédire la classe de la transaction.
    """)
    st.image("detection_fraude.jpeg")

# Exploration des données
elif page == pages[1]:
    st.write("### Exploration des données")
    st.write("Voici un aperçu des données utilisées dans ce projet :")
    st.dataframe(df_creditcard.head())
    st.write(f"Dimensions du dataset : {df_creditcard.shape}")
    st.write(f"Données par classe : {df_creditcard['Class'].value_counts()}")

    if st.checkbox("Afficher les valeurs manquantes"): 
        st.write(df_creditcard.isna().sum())
        
    if st.checkbox("Afficher les doublons"): 
        st.write("Nombre de doublons :", df_creditcard.duplicated().sum())
    
    if st.checkbox("Afficher le résumé Statistique des données"): 
        st.write("Résumé statistique des données :", df_creditcard.describe())


# Analyse de données 
elif page == pages[2]:
    st.write("### Analyse de données")
    
    # Distribution de la variable cible
    st.write("#### Distribution de la variable cible (Class)")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Class', data=df_creditcard, ax=ax1, palette="coolwarm")
    ax1.set_title("Distribution des classes")
    st.pyplot(fig1)

    # Distribution des Montants des Transactions selon la Classe
    afficher_outliers = st.sidebar.checkbox("Afficher les valeurs aberrantes montant", value=False)
    st.write("### Distribution des Montants par Classe")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.boxplot(
    x='Class', 
    y='Amount', 
    data=df_creditcard, 
    palette='Set2',
    showfliers=afficher_outliers, 
    ax=ax2
    )
    ax2.set_title('Distribution des Montants des Transactions selon la Classe')
    ax2.set_xlabel('Classe (0 = Non-Fraude, 1 = Fraude)')
    ax2.set_ylabel('Montant')
    ax2.set_xticklabels(['Non-Fraude', 'Fraude'])
    st.pyplot(fig2)


    # Distribution des Montants des Transactions selon la Classe
    st.write("### Distribution du temps des Transactions par Classe")
    afficher_outlierst = st.sidebar.checkbox("Afficher les valeurs aberrantes temps", value=False)
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.boxplot(
    x='Class', 
    y='Time', 
    data=df_creditcard, 
    palette='coolwarm',
    showfliers=afficher_outlierst, 
    ax=ax3
    )
    ax3.set_title('Distribution des Montants des Transactions selon la Classe')
    ax3.set_xlabel('Classe (0 = Non-Fraude, 1 = Fraude)')
    ax3.set_ylabel('Temps')
    ax3.set_xticklabels(['Non-Fraude', 'Fraude'])
    st.pyplot(fig3)

    # Matrice de corrélation
    st.write("### Matrice de corrélation")
    matrice_de_correlation = df_creditcard.corr()
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(matrice_de_correlation, dtype=bool))
    sns.heatmap(matrice_de_correlation, annot=False, cmap='coolwarm', mask=mask, ax=ax4)
    st.pyplot(fig4)

    # Visualisation interactive avec Plotly
    st.write("### Visualisation interactive")
    fig4 = px.scatter(
        df_creditcard, x="Time", y="Amount", color="Class", 
        title="Montant des transactions dans le temps",
        labels={"Amount": "Montant", "Time": "Temps"}
    )
    st.plotly_chart(fig4)

# Modélisation
elif page == pages[3]:
    st.write("### Modélisation")
    
    # Séparation des données
    X = df_creditcard.drop(columns=['Class'], axis=1)
    y = df_creditcard['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Chargement des modèles sauvegardés
    reg_log = joblib.load("modele_regression_logistique.pkl")
    rf = joblib.load("optimized_random_forest.pkl")
    xgb = joblib.load("xgboost_optimise.pkl")
    
    # Prédictions
    model_choisi = st.selectbox(label="Choisissez un modèle :", options=["Régression Logistique", "Random Forest", "XGBoost"])
    
    if model_choisi == "Régression Logistique":
        y_pred = reg_log.predict(X_test)
        proba = reg_log.predict_proba(X_test)[:, 1]
    elif model_choisi == "Random Forest":
        y_pred = rf.predict(X_test)
        proba = rf.predict_proba(X_test)[:, 1]
    elif model_choisi == "XGBoost":
        y_pred = xgb.predict(X_test)
        proba = xgb.predict_proba(X_test)[:, 1]

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    st.write("#### Matrice de Confusion :")
    st.write(cm)

    # Rapport de classification
    st.write("#### Rapport de Classification :")
    st.text(classification_report(y_test, y_pred, target_names=["Non-Fraude", "Fraude"]))

    # Courbe ROC
    st.write("#### Courbe ROC")
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_score = auc(fpr, tpr)
    fig4, ax4 = plt.subplots()
    ax4.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    ax4.set_title("Courbe ROC")
    ax4.set_xlabel("Taux de Faux Positifs")
    ax4.set_ylabel("Taux de Vrais Positifs")
    ax4.legend(loc="best")
    st.pyplot(fig4)

# === Pied de page ===
st.markdown("<p style='text-align: center;'>Développé avec ❤️ en Streamlit par Tebatto</p>", unsafe_allow_html=True)
