"""Run offline classification, metrics and the FastAPI test client."""

from fastapi.testclient import TestClient
from document_ai.api import create_app
from document_ai.classification import train_classifier
from document_ai.examples import TRAIN_LABELS, TRAIN_TEXTS
from document_ai.retrieval import retrieval_metrics

model = train_classifier(TRAIN_TEXTS, TRAIN_LABELS)
print(TestClient(create_app(model)).post("/predict", json={"text": "invoice amount payment"}).json())
print(retrieval_metrics(["wrong", "invoice_demo"], {"invoice_demo"}, k=2))
