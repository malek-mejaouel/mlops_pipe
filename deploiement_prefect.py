"""Déploiements Prefect - planification quotidienne des flows (Atelier 3, étape 05).

Ce script enregistre les flows du projet comme *deployments* planifiés une
fois par jour, puis démarre un worker en local pour les exécuter.

Mise en route (3 terminaux) :

    Terminal 1 - le serveur (base de données + UI sur http://localhost:4200)
        prefect server start

    Terminal 2 - le worker (enregistre et exécute les flows planifiés)
        prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
        python deploiement_prefect.py

    Terminal 3 - le déclencheur (déclenchement manuel d'un déploiement)
        prefect deployment ls
        prefect deployment run 'all/ml-pipeline-all'
        prefect deployment run 'train/ml-pipeline-train'

Sans server (terminal 1) : pas d'UI ni de coordination.
Sans worker (terminal 2) : les flows sont enregistrés mais jamais exécutés.
Sans déclencheur (terminal 3) : aucun lancement manuel.
"""

from prefect import serve

from pipeline_prefect import all_flow, code_flow, evaluate_flow, train_flow

# Planification : tous les jours à 02h00 (syntaxe cron Prefect).
DAILY_AT_2AM = "0 2 * * *"


if __name__ == "__main__":
    serve(
        all_flow.to_deployment(name="ml-pipeline-all", cron=DAILY_AT_2AM),
        train_flow.to_deployment(name="ml-pipeline-train", cron=DAILY_AT_2AM),
        evaluate_flow.to_deployment(name="ml-pipeline-evaluate", cron=DAILY_AT_2AM),
        code_flow.to_deployment(name="ml-pipeline-code", cron=DAILY_AT_2AM),
    )
