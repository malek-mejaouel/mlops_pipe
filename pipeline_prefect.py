"""Orchestration Prefect du pipeline Customer Churn - Atelier 3.

Ce module regroupe les différentes étapes du projet sous forme de *tasks* et
de *flows* Prefect afin de simplifier et d'automatiser leur exécution.

Tasks :
    1. retrieve_project     : récupérer le projet depuis le dépôt git distant.
    2. install_dependencies : installer les dépendances Python.
    3. format_code          : formater le code (black).
    4. lint_code            : vérifier la qualité du code (flake8).
    5. security_code        : analyser la sécurité du code (bandit).
    6. run_unit_tests       : exécuter les tests unitaires (pytest).
    7. prepare_data         : préparer les données (main.py).
    8. train_model          : entraîner le modèle (main.py).
    9. save_model           : sauvegarder le modèle (main.py).
   10. load_model           : charger le modèle (main.py).
   11. evaluate_model       : évaluer le modèle (main.py).

Flows :
    - code     : étapes 3 à 6 (suivi de la qualité du code).
    - train    : préparation des données + entraînement + sauvegarde.
    - evaluate : chargement du modèle + évaluation.
    - all      : récupération + dépendances + code + pipeline complet.

Utilisation :
    python pipeline_prefect.py --flow all
    python pipeline_prefect.py --flow train
    python pipeline_prefect.py --flow evaluate
    python pipeline_prefect.py --flow code
"""

import argparse
import subprocess
import sys

from prefect import flow, task

from main import (
    evaluate_model as _evaluate_model,
    load_model as _load_model,
    prepare_data as _prepare_data,
    save_model as _save_model,
    train_model as _train_model,
)

# Fichiers ciblés par les étapes de qualité et de sécurité.
# On limite volontairement l'analyse à ces fichiers pour éviter de scanner
# tout le dépôt (notamment le dossier venv) et de ralentir le pipeline.
TARGET_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]

# Nom du fichier de tests unitaires exécuté par la task run_unit_tests.
TEST_FILE = "test_model_pipeline.py"

# Dépôt git distant du projet (étape 06).
REPO_URL = "https://github.com/malek-mejaouel/mlops_pipe.git"


def _run(command):
    """Exécuter une commande et retourner (code de retour, sortie fusionnée)."""
    result = subprocess.run(command, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    print(f"$ {' '.join(command)}\n{output}".rstrip())
    return result.returncode, output


# --- Tasks liées au code ----------------------------------------------------


@task
def retrieve_project():
    """Récupérer le projet depuis le dépôt git distant (étape 06).

    Si le dossier courant n'est pas encore un dépôt git, le projet est cloné
    depuis ``REPO_URL`` (récupération sur une nouvelle machine). Sinon, la
    dernière version est récupérée avec ``git pull``.

    Entrée  : aucune.
    Sortie  : code de retour de la commande git (0 en cas de succès).
    """
    is_repo = (
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )
    if is_repo:
        code, _ = _run(["git", "pull", "--ff-only"])
    else:
        code, _ = _run(["git", "clone", REPO_URL, "."])

    if code != 0:
        print("Avertissement : récupération du projet impossible.")
    return code


@task
def install_dependencies():
    """Installer les dépendances Python listées dans ``requirements.txt``.

    Entrée  : aucune.
    Sortie  : aucune (lève une erreur si l'installation échoue).
    """
    code, _ = _run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    if code != 0:
        raise RuntimeError("Échec de l'installation des dépendances.")


@task
def format_code():
    """Formater le code source du projet avec ``black``.

    Entrée  : aucune.
    Sortie  : code de retour de black.
    """
    return _run([sys.executable, "-m", "black", *TARGET_FILES])[0]


@task
def lint_code():
    """Vérifier la qualité du code avec ``flake8``.

    Entrée  : aucune.
    Sortie  : code de retour de flake8.
    """
    return _run([sys.executable, "-m", "flake8", *TARGET_FILES])[0]


@task
def security_code():
    """Analyser la sécurité du code avec ``bandit``.

    Entrée  : aucune.
    Sortie  : code de retour de bandit.
    """
    return _run([sys.executable, "-m", "bandit", *TARGET_FILES])[0]


@task
def run_unit_tests():
    """Exécuter les tests unitaires avec ``pytest``.

    Entrée  : aucune.
    Sortie  : code de retour de pytest.
    """
    return _run([sys.executable, "-m", "pytest", "-q", TEST_FILE])[0]


# --- Tasks liées au modèle --------------------------------------------------


@task
def prepare_data(data_path="Churn_Modelling.csv"):
    """Préparer les données via ``main.prepare_data``.

    Entrée  : chemin du fichier CSV.
    Sortie  : x_train, x_test, y_train, y_test.
    """
    return _prepare_data(data_path)


@task
def train_model(x_train, y_train):
    """Entraîner le modèle via ``main.train_model``.

    Entrée  : features et cible d'entraînement.
    Sortie  : modèle entraîné.
    """
    return _train_model(x_train, y_train)


@task
def save_model(model, model_path="classifier.joblib"):
    """Sauvegarder le modèle via ``main.save_model``.

    Entrée  : modèle entraîné, chemin de destination.
    Sortie  : chemin du fichier écrit.
    """
    return _save_model(model, model_path)


@task
def load_model(model_path="classifier.joblib"):
    """Charger le modèle via ``main.load_model``.

    Entrée  : chemin du fichier modèle.
    Sortie  : modèle chargé.
    """
    return _load_model(model_path)


@task
def evaluate_model(model, x_test, y_test):
    """Évaluer le modèle via ``main.evaluate_model``.

    Entrée  : modèle entraîné, features et cible de test.
    Sortie  : dictionnaire de métriques (accuracy, matrice, rapport...).
    """
    return _evaluate_model(model, x_test, y_test)


# --- Flows ------------------------------------------------------------------


@flow(name="code")
def code_flow():
    """Flow de suivi de la qualité du code (formatage, qualité, sécurité, tests)."""
    format_code()
    lint_code()
    security_code()
    run_unit_tests()


@flow(name="train")
def train_flow():
    """Flow d'entraînement : préparation des données, entraînement, sauvegarde."""
    x_train, x_test, y_train, y_test = prepare_data()
    model = train_model(x_train, y_train)
    save_model(model)
    return model


@flow(name="evaluate")
def evaluate_flow():
    """Flow d'évaluation : chargement du modèle sauvegardé puis évaluation."""
    x_train, x_test, y_train, y_test = prepare_data()
    model = load_model()
    return evaluate_model(model, x_test, y_test)


@flow(name="all")
def all_flow():
    """Flow complet : récupération, dépendances, qualité du code, puis pipeline ML."""
    retrieve_project()
    install_dependencies()
    code_flow()

    x_train, x_test, y_train, y_test = prepare_data()
    model = train_model(x_train, y_train)
    save_model(model)
    return evaluate_model(model, x_test, y_test)


FLOWS = {
    "all": all_flow,
    "train": train_flow,
    "evaluate": evaluate_flow,
    "code": code_flow,
}


def main():
    """Analyser les arguments CLI et lancer le flow demandé."""
    parser = argparse.ArgumentParser(
        description="Pipelines Prefect - Customer Churn (Atelier 3)"
    )
    parser.add_argument(
        "--flow",
        choices=list(FLOWS),
        default="all",
        help="Flow à exécuter.",
    )
    args = parser.parse_args()
    FLOWS[args.flow]()


if __name__ == "__main__":
    main()
