from fastapi.testclient import TestClient
from document_ai.api import create_app
from document_ai.classification import error_analysis, evaluate_classifier, train_classifier
from document_ai.evaluation import exact_match, success_rate
from document_ai.examples import TEST_LABELS, TEST_TEXTS, TRAIN_LABELS, TRAIN_TEXTS
from document_ai.retrieval import reciprocal_rank_fusion, retrieval_metrics


def test_classifier():
    model = train_classifier(TRAIN_TEXTS, TRAIN_LABELS)
    assert model.predict(["invoice amount payment"])[0] == "invoice"
    # Held-out sentences, not the training data
    assert evaluate_classifier(model, TEST_TEXTS, TEST_LABELS)["accuracy"] >= 0.75


def test_error_analysis():
    report = error_analysis(train_classifier(TRAIN_TEXTS, TRAIN_LABELS), TEST_TEXTS, TEST_LABELS)
    assert sum(map(sum, report["confusion_matrix"])) == len(TEST_TEXTS)
    assert len(report["errors"]) == len(TEST_TEXTS) - sum(report["confusion_matrix"][i][i] for i in range(2))


def test_api():
    client = TestClient(create_app(train_classifier(TRAIN_TEXTS, TRAIN_LABELS)))
    response = client.post("/predict", json={"text": "invoice amount payment"})
    assert response.status_code == 200
    assert response.json() == {"category": "invoice"}


def test_retrieval_metrics():
    scores = retrieval_metrics(["wrong", "right"], {"right"}, k=2)
    assert scores["recall_at_k"] == 1.0
    assert scores["precision_at_k"] == 0.5
    assert scores["reciprocal_rank"] == 0.5


def test_evaluation():
    assert exact_match(" EUR 250 ", "eur 250")
    assert success_rate([True, False, True]) == 2 / 3


def test_reciprocal_rank_fusion():
    assert reciprocal_rank_fusion([["a", "b"], ["b", "c"], ["b", "a"]]) == ["b", "a", "c"]
