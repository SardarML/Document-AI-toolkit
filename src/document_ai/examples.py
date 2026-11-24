"""Synthetic samples for demonstration only; not benchmark data."""

TRAIN_TEXTS = [
    "Invoice payment due date", "Invoice number total tax amount",
    "Billing statement invoice payment", "Service agreement between the parties",
    "Contract term termination clause", "Agreement signed by both parties governing law",
]
TRAIN_LABELS = ["invoice"] * 3 + ["contract"] * 3

TEST_TEXTS = [
    "Please pay the invoice total", "Contract renewal and termination notice",
    "Payment due tax invoice number", "The parties agree to the following terms",
]
TEST_LABELS = ["invoice", "contract", "invoice", "contract"]

SAMPLE_DOCUMENTS = {
    "invoice_demo": "Invoice INV-100. Total: EUR 250. Due date: 2026-11-15.",
    "contract_demo": "Service agreement between Example Ltd and Demo GmbH. Term: 12 months. Notice period: 30 days.",
}

GOLDEN = [
    {"question": "What is the invoice total?", "relevant": {"invoice_demo"}, "reference": "EUR 250"},
    {"question": "What is the notice period?", "relevant": {"contract_demo"}, "reference": "30 days"},
]
