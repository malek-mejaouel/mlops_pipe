"""Point d'entrée CLI du pipeline Machine Learning Customer Churn.

Exécute les différentes étapes de ``model_pipeline.py`` via des arguments
en ligne de commande.

Exemples d'utilisation :
    python main.py --step all
    python main.py --step prepare
    python main.py --step train
    python main.py --step evaluate --model classifier.joblib
    python main.py --step load
"""

import argparse

from model_pipeline import (
    DATA_PATH,
    MODEL_PATH,
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def run_pipeline(data_path=DATA_PATH, model_path=MODEL_PATH):
    """Exécuter le pipeline complet : préparation, entraînement, évaluation, sauvegarde.

    Entrée  : chemin des données et chemin de sauvegarde du modèle.
    Sortie  : modèle entraîné.
    """
    print("[1/4] Préparation des données...")
    x_train, x_test, y_train, y_test = prepare_data(data_path)

    print("[2/4] Entraînement du modèle...")
    model = train_model(x_train, y_train)

    print("[3/4] Évaluation du modèle...")
    evaluate_model(model, x_test, y_test)

    print("[4/4] Sauvegarde du modèle...")
    save_model(model, model_path)
    print(f"Modèle sauvegardé dans : {model_path}")

    return model


def main():
    """Analyser les arguments CLI et exécuter l'étape demandée."""
    parser = argparse.ArgumentParser(
        description="Pipeline ML modulaire - Customer Churn"
    )
    parser.add_argument(
        "--data", default=DATA_PATH, help="Chemin du fichier CSV des données."
    )
    parser.add_argument(
        "--model", default=MODEL_PATH, help="Chemin du modèle à sauvegarder/charger."
    )
    parser.add_argument(
        "--step",
        choices=["all", "prepare", "train", "evaluate", "load"],
        default="all",
        help="Étape à exécuter.",
    )
    args = parser.parse_args()

    if args.step == "all":
        run_pipeline(args.data, args.model)

    elif args.step == "prepare":
        x_train, x_test, y_train, y_test = prepare_data(args.data)
        print(f"X_train: {x_train.shape}, X_test: {x_test.shape}")
        print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

    elif args.step == "train":
        x_train, x_test, y_train, y_test = prepare_data(args.data)
        model = train_model(x_train, y_train)
        save_model(model, args.model)
        print(f"Modèle entraîné et sauvegardé dans : {args.model}")

    elif args.step == "evaluate":
        x_train, x_test, y_train, y_test = prepare_data(args.data)
        model = load_model(args.model)
        evaluate_model(model, x_test, y_test)

    elif args.step == "load":
        model = load_model(args.model)
        print(f"Modèle chargé : {model}")


if __name__ == "__main__":
    main()
