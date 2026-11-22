import pytest
from document_ai.classification import train_classifier
from document_ai.examples import TRAIN_LABELS, TRAIN_TEXTS


def test_workflow():
    pytest.importorskip("langgraph")
    from document_ai.workflow import build_workflow

    graph = build_workflow(train_classifier(TRAIN_TEXTS, TRAIN_LABELS))
    assert graph.invoke({"text": "invoice amount payment"})["category"] == "invoice"
