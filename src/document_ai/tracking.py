"""Optional MLflow experiment logging."""


def log_evaluation(model, texts: list[str], labels: list[str], experiment: str = "Document AI"):
    """Log a model configuration and evaluation metrics to MLflow."""
    import mlflow
    from .classification import evaluate_classifier

    scores = evaluate_classifier(model, texts, labels)
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("classifier", "LogisticRegression")
        mlflow.log_metrics(scores)
    return scores
