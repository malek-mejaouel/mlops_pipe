"""Tests unitaires du pipeline Customer Churn (model_pipeline.py).

Ces tests sont exécutés par la task Prefect ``run_unit_tests`` (via pytest).
"""

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def test_prepare_data_shapes():
    """Vérifier la cohérence des dimensions après préparation des données."""
    x_train, x_test, y_train, y_test = prepare_data()

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert x_train.shape[1] == x_test.shape[1]


def test_train_and_evaluate():
    """Vérifier que le modèle s'entraîne et retourne des métriques valides."""
    x_train, x_test, y_train, y_test = prepare_data()

    model = train_model(x_train, y_train, n_estimators=10, random_state=1)
    results = evaluate_model(model, x_test, y_test)

    assert 0.0 <= results["accuracy"] <= 1.0
    assert len(results["y_pred"]) == len(y_test)


def test_save_and_load_model(tmp_path):
    """Vérifier que le modèle sauvegardé peut être rechargé."""
    x_train, _, y_train, _ = prepare_data()

    model = train_model(x_train, y_train, n_estimators=10, random_state=1)
    model_path = tmp_path / "model.joblib"

    save_model(model, str(model_path))
    loaded = load_model(str(model_path))

    assert hasattr(loaded, "predict")
