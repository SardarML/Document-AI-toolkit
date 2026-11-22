"""Simple API factory with an explicitly supplied classifier."""

from fastapi import FastAPI
from pydantic import BaseModel


class PredictRequest(BaseModel):
    text: str


def create_app(classifier) -> FastAPI:
    """Expose document classification as POST /predict."""
    app = FastAPI(title="Document AI Toolkit")

    @app.post("/predict")
    def predict(body: PredictRequest) -> dict:
        return {"category": str(classifier.predict([body.text])[0])}

    return app
