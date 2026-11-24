"""Retrieval pipeline and ranked retrieval evaluation."""

import math
import re


def build_retriever(sources: dict[str, str], k: int = 3, chunk_size: int = 500):
    """Create an in-memory retriever using local multilingual embeddings."""
    from langchain_core.documents import Document
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = [Document(page_content=text, metadata={"source": name})
            for name, text in sources.items() if text.strip()]
    if not docs:
        raise ValueError("At least one nonempty document is required")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=min(80, chunk_size // 5)
    ).split_documents(docs)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}-{index}"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return InMemoryVectorStore.from_documents(chunks, embeddings).as_retriever(
        search_kwargs={"k": k}
    )


def retrieval_metrics(ranked: list[str], relevant: set[str], k: int = 3) -> dict:
    """Return precision/recall/nDCG at k and reciprocal rank."""
    if k < 1:
        raise ValueError("k must be positive")
    top = ranked[:k]
    hits = [int(item in relevant) for item in top]
    first = next((i for i, item in enumerate(ranked, 1) if item in relevant), None)
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return {
        "precision_at_k": sum(hits) / k,
        "recall_at_k": sum(hits) / len(relevant) if relevant else 0.0,
        "reciprocal_rank": 1 / first if first else 0.0,
        "ndcg_at_k": dcg / ideal if ideal else 0.0,
    }


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Fuse ranked ID lists; the highest fused score comes first."""
    fused = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, 1):
            fused[item] = fused.get(item, 0) + 1 / (k + position)
    return sorted(fused, key=fused.get, reverse=True)


def hybrid_search(query: str, docs: list, store, k: int = 3) -> list[str]:
    """Combine BM25 and vector rankings using reciprocal rank fusion."""
    import numpy as np
    from rank_bm25 import BM25Okapi

    if not docs:
        return []
    tokenize = lambda value: re.findall(r"\w+", value.casefold())
    scores = BM25Okapi([tokenize(d.page_content) for d in docs]).get_scores(tokenize(query))
    lexical = [docs[i].metadata["chunk_id"] for i in np.argsort(scores)[::-1]]
    semantic = [d.metadata["chunk_id"] for d in store.similarity_search(query, k=len(docs))]
    return reciprocal_rank_fusion([lexical, semantic])[:k]
