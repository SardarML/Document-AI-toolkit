# Document AI Toolkit

A small, modular **Python reference project** for image OCR, document classification, named entity recognition (NER), retrieval-augmented generation (RAG), LangGraph workflows and agents, in-process MCP tools, testing and experiment tracking.

It is intended for **learning, prototyping and technical demonstrations**, not as a production-ready all-purpose document processing platform.

## Architecture

```text
Document image -> PaddleOCR -> Text ─┬─> TF-IDF classifier -> Category
                                    ├─> spaCy NER -> Entities
                                    ├─> LangChain chunking + embeddings -> Retrieval -> LLM answer
                                    └─> LangGraph workflow / tool agent -> optional MCP read tool
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
python examples/quickstart.py
pytest -q
```

### Gradio demo

```bash
pip install -e '.[all]'
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm   # optional for German
export OPENAI_API_KEY='your-api-key'      # optional for RAG/agent; use a secrets manager
python -m document_ai.ui
```

The first PaddleOCR and sentence-transformer calls download model weights. An LLM API key is required for generation and the agent, but **not** for classification, local retrieval or basic tests. Model downloads and OpenAI requests may incur costs.

The UI uses **per-session document state** rather than a shared global document list. Do not publicly share the demo with sensitive documents; verify the hosting environment, access controls, retention and provider terms first.

### Use modules independently

```python
from document_ai.classification import train_classifier
from document_ai.retrieval import retrieval_metrics

model = train_classifier(
    ["payment invoice", "invoice number", "patient diagnosis", "clinical note"],
    ["invoice", "invoice", "medical_record", "medical_record"],
)
print(model.predict(["invoice payment"]))
print(retrieval_metrics(["b", "a"], {"a"}, k=2))
```

### In-process MCP (no HTTP port)

```python
import asyncio
from document_ai.mcp_tools import make_mcp, read_mcp

async def demo():
    server = make_mcp({"doc1": "Synthetic example text"})
    print(await read_mcp(server, "doc1"))

asyncio.run(demo())
```

In Jupyter, replace `asyncio.run(demo())` with `await demo()`.

### FastAPI

The `create_app(classifier)` factory exposes `POST /predict`. A real deployment should load a persisted, validated model rather than training at startup. Tests use `fastapi.testclient.TestClient`, so no server is required.

### Evaluation

- **Classification:** accuracy, macro precision, macro recall and macro F1 (`classification.py`)
- **Retrieval:** Precision@k, Recall@k, reciprocal rank and nDCG@k (`retrieval.py`)
- **Generation:** strict exact-match baseline (`evaluation.py`); semantic correctness, faithfulness and answer relevance require human review or a validated judge
- **Agents:** task and tool success rates require labeled execution traces; this repository demonstrates a success-rate helper
- **Operations:** optional MLflow logging (`tracking.py`), pytest and GitHub Actions CI

**Golden datasets:** `examples.py` contains synthetic demonstration cases, not independently annotated benchmark measurements. Evaluate retrieval and generation separately using real, representative ground truth before reporting performance.

## Important limitations

- The six classifier training sentences are deliberately tiny and cannot support credible generalization claims.
- `read_table` from the original Colab prototype was layout-dependent; general table reconstruction is not included.
- Standard spaCy models do **not** reliably identify medical conditions or other domain-specific entities.
- In-memory vector storage has no persistence, authentication or multi-worker synchronization. The Gradio demo rebuilds a local index per question for clarity, which is inefficient at scale.
- The MCP demo runs inside one Python process. It does not expose remote MCP transport.
- Prompt-injection resistance is **not guaranteed** by system prompts or marking text as untrusted. Use permissions, validation and sandboxing in real deployments.
- Agent tools are read-only. Do not expose write-capable tools without authorization and idempotency protections.
- The example `langgraph` workflow classifies and routes; it **does not** perform domain-specific field extraction.
- Optional OCR/NER/RAG/MCP dependencies and live hosted API calls have not all been integration-tested together. Run the demo in your target environment and pin known-good versions before production.

## Repository structure

```text
src/document_ai/   # Independent modules
examples/          # Offline quickstart
 tests/            # Offline unit tests plus optional integration tests
.github/workflows/ # Continuous integration
```

## Security and licensing

Never commit `.env`, API credentials, client documents, user data or proprietary algorithms. Use synthetic data unless you have permission and appropriate safeguards. Before publishing, verify ownership and licensing of the code, model weights, datasets and any reused material.

MIT license applies to the code in this repository, **not** to third-party models or datasets.
