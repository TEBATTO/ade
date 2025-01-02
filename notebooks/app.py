# Importation des libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Titre de l'application
st.title("Tableau de Bord Interactif pour l’Analyse des Données")
st.write("Ce tableau de bord permet d'explorer les données et de visualiser les tendances.")

# Chargement des données avec cache
@st.cache_data
def load_data():
    data = pd.read_csv("creditcard.csv")
    return data

data = load_data()
st.write(data.head())  # Affiche les premières lignes

# Distribution de la variable Time
st.subheader("Distribution de la variable Time")
fig, ax = plt.subplots()
ax.hist(data["Time"], bins=50)
st.pyplot(fig)

# Distribution de la variable Amount
st.subheader("Distribution de la variable Amount")
fig, ax = plt.subplots()
ax.hist(data["Amount"], bins=50)
st.pyplot(fig)

# Visualisation interactive avec Plotly
st.subheader("Variation de Amount en fonction de Time")
fig = px.line(data, x="Time", y="Amount")
st.plotly_chart(fig)

# Filtrage par catégorie
options = st.selectbox("Choisissez une catégorie", data["Time"].unique())
data_filtered = data[data["Time"] == options]
st.write(data_filtered)

# Slider pour filtrer la plage de valeurs de Amount
valeur_min, valeur_max = st.slider("Choisissez la plage de valeurs", 
                                   min_value=int(data["Amount"].min()), 
                                   max_value=int(data["Amount"].max()), 
                                   value=(10, 50))
data_filtered = data[(data["Amount"] >= valeur_min) & (data["Amount"] <= valeur_max)]
st.write(data_filtered)

# Afficher les statistiques descriptives
if st.checkbox("Afficher les statistiques descriptives"):
    st.write(data.describe())

# Matrice de corrélation entre Time et Amount
st.subheader("Matrice de corrélation")
fig, ax = plt.subplots(figsize=(8, 8))
corr_matrix = data[['Time', 'Amount']].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Ajout des modèles d'apprentissage
st.subheader("Modèles d'apprentissage")

# Préparation des données pour les modèles
X = data.drop(columns='Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Choix des modèles
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier()
}

# Boucle pour entraîner les modèles, afficher les matrices de confusion et les rapports de classification
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.subheader(f"Résultats pour {model_name}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Matrice de confusion pour {model_name}")
    st.pyplot(fig)
    
    # Rapport de classification
    st.write("Rapport de classification:")
    st.text(classification_report(y_test, y_pred))

# Onglet Graphes interactif
st.sidebar.title("Graphes Interactifs")
st.sidebar.write("Sélectionnez le type de graphique et la variable à visualiser.")

# Options de graphiques et de variables
chart_type = st.sidebar.selectbox("Type de graphique", ["Boxplot", "Histogramme", "Camembert"])
variable = st.sidebar.selectbox("Variable", data.columns)

if chart_type == "Boxplot":
    fig = px.box(data, y=variable)
    st.plotly_chart(fig)
elif chart_type == "Histogramme":
    fig = px.histogram(data, x=variable)
    st.plotly_chart(fig)
elif chart_type == "Camembert":
    if data[variable].dtype == 'object' or len(data[variable].unique()) < 10:
        fig = px.pie(data, names=variable)
        st.plotly_chart(fig)
    else:
        st.write("Le camembert est uniquement pour les variables catégorielles ou ayant peu de valeurs uniques.")
