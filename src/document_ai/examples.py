"""Synthetic samples for demonstration only; not benchmark data."""

TRAIN_TEXTS = [
    "Patient diagnosis treatment consultation", "Clinical examination discharge summary",
    "Patient medication clinical assessment", "Invoice payment due date",
    "Invoice number total tax amount", "Billing statement invoice payment",
]
TRAIN_LABELS = ["medical_record"] * 3 + ["invoice"] * 3

SAMPLE_DOCUMENTS = {
    "invoice_demo": "Invoice INV-100. Total: EUR 250. Due date: 2026-11-15.",
    "medical_demo": "Synthetic medical note. Diagnosis: hypertension. Follow-up: December.",
}

GOLDEN = [
    {"question": "What is the invoice total?", "relevant": {"invoice_demo"}, "reference": "EUR 250"},
    {"question": "What is the diagnosis?", "relevant": {"medical_demo"}, "reference": "hypertension"},
]
