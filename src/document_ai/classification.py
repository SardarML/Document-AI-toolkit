"""TF-IDF document classification with scikit-learn."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def train_classifier(texts: list[str], labels: list[str]):
    """Train a classifier on labeled document texts."""
    if len(texts) != len(labels) or len(set(labels)) < 2:
        raise ValueError("Provide matching texts and labels from at least two classes")
    return make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(texts, labels)


def evaluate_classifier(model, texts: list[str], labels: list[str]) -> dict:
    """Compute out-of-sample classification metrics."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    predictions = model.predict(texts)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def error_analysis(model, texts: list[str], labels: list[str]) -> dict:
    """Return the confusion matrix and the misclassified examples."""
    from sklearn.metrics import confusion_matrix

    predictions = model.predict(texts)
    classes = [str(c) for c in model.classes_]
    return {
        "classes": classes,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
        "errors": [(t, l, p) for t, l, p in zip(texts, labels, predictions) if l != p],
    }
