"""Pipeline Machine Learning modulaire pour le projet Customer Churn.

Ce module regroupe sous forme de fonctions réutilisables les différentes
étapes du script Jupyter ``customer_churn.ipynb`` :

    1. prepare_data()  : charger, nettoyer et découper les données.
    2. train_model()   : entraîner le modèle de classification.
    3. evaluate_model(): évaluer les performances du modèle.
    4. save_model()    : sauvegarder le modèle entraîné avec joblib.
    5. load_model()    : recharger un modèle sauvegardé.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Constantes du pipeline -------------------------------------------------
DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"
TARGET = "Exited"
DROP_COLUMNS = ["Surname", "Geography", "RowNumber", "CustomerId"]


def load_data(data_path=DATA_PATH):
    """Charger le jeu de données CSV dans un DataFrame pandas.

    Entrée  : chemin du fichier CSV.
    Sortie  : DataFrame contenant toutes les colonnes brutes.
    """
    return pd.read_csv(data_path)


def prepare_data(data_path=DATA_PATH, test_size=0.2, random_state=1):
    """Charger, nettoyer et découper les données en train / test.

    Étapes : encodage de la variable catégorielle ``Gender``, suppression des
    colonnes inutiles, séparation des features (X) et de la cible (y), puis
    découpage train / test.

    Entrée  : chemin du CSV, taille du jeu de test, graine aléatoire.
    Sortie  : X_train, X_test, y_train, y_test.
    """
    df = load_data(data_path)

    # Encodage de la variable catégorielle 'Gender' (0 = Female, 1 = Male).
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # Suppression des colonnes non pertinentes (identifiants, texte libre).
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    # Séparation des features et de la variable cible.
    x = df.drop(columns=[TARGET])
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train, n_estimators=100, random_state=42):
    """Entraîner un classifieur RandomForest sur les données d'entraînement.

    Entrée  : features et cible d'entraînement, hyperparamètres du modèle.
    Sortie  : modèle entraîné.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Évaluer les performances du modèle sur le jeu de test.

    Entrée  : modèle entraîné, features et cible de test.
    Sortie  : dictionnaire contenant l'accuracy, la matrice de confusion,
              le rapport de classification et les prédictions.
    """
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy score: {accuracy * 100:.2f}%")
    print("Confusion matrix:\n", matrix)
    print("Classification report:\n", report)

    return {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "report": report,
        "y_pred": y_pred,
    }


def save_model(model, model_path=MODEL_PATH):
    """Sauvegarder le modèle entraîné sur le disque avec joblib.

    Entrée  : modèle entraîné, chemin de destination.
    Sortie  : chemin du fichier écrit.
    """
    joblib.dump(model, model_path)
    return model_path


def load_model(model_path=MODEL_PATH):
    """Charger un modèle préalablement sauvegardé avec joblib.

    Entrée  : chemin du fichier modèle.
    Sortie  : modèle chargé.
    """
    return joblib.load(model_path)
